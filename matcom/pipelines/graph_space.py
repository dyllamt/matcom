import matcom.tools.edge_calculators as edg
import matcom.tools.dist_calculators as dist

import numpy as np

from matcom.pipelines.structure_space import FRAMEWORK_FEATURIZER

from dataspace.base import Pipe, in_batches
from dataspace.workspaces.local_db import MongoFrame

from pymatgen.core import Structure
from pymatgen.analysis.defects.generators import VacancyGenerator

from pandas import DataFrame

'''
this module implements a pipeline for constructing graphs from feature spaces
'''


class GenerateGraphCollection(Pipe):
    '''
    structures (verticies) within a similarity threshold are connected by edges
    to form a graph of the structure space. additional edges connect structures
    that are similar to another when a defect (strain/vacancy/interstical) is
    introduced to one of the structures (e.g. rocksalt + intersticial = BCC)

    the graph is stored as an adjacency list to conserve storage/memory
    '''
    def __init__(self, path='/data/db', database='structure_graphs',
                 structure_collection='structure', graph_collection='graph'):
        '''
        Args:
            path (str) path to a local mongodb directory
            database (str) name of a pymongo database
            structure_collection (str) name of a pymongo collection that holds
                data on the structures being analyzed
            graph_collection (str) name of a pymongo collection that holds data
                on the graph representation of the structures
        '''
        structure_space = MongoFrame(
            path=path, database=database, collection=structure_collection)
        graph_space = MongoFrame(
            path=path, database=database, collection=graph_collection)
        Pipe.__init__(self, source=structure_space, destination=graph_space)

    def populate_verticies(self,
                           criteria={'structure_features': {'$exists': True},
                                     'nsites': {'$lte': 51}}):
        '''
        populate verticies in graph space with verticies from structure space
        '''
        self.source.from_storage(filter=criteria,
                                 projection={'material_id': 1})
        self.transfer(to='destination')
        self.destination.to_storage(identifier='material_id', upsert=True)

    @in_batches
    def find_edges(self, threshold=0.5, batch_size=10000,
                   edge_calculator=edg.sub_pairwise_squared_similarity):
        '''
        solve for undirected, boolean edges in the structure space. the graph
        structure is stored as an adjacency list using this document schema:
            'material_id' : str : the source vertex
            'edges' : list of str : the destination verticies

        Args:
            threshold (float) distance threshold to connect an edge
            batch_size (int) batch size for computing pairwise distances when
                generating graph edges. subject to memory constraints
            edge_calculator (func) a sub-pairwise distance calculator that
                returns an N x M adjacency matrix
        '''

        # load material ids without defined edges
        self.destination.from_storage(filter={'edges':
                                              {'$exists': False}},
                                      projection={'material_id': 1},
                                      limit=batch_size)

        if len(self.destination.memory.index) == 0:

            return 0  # return False when update is complete

        else:

            # save names of verticies
            verticies = self.destination.memory['material_id'].values
            self.destination.memory = None  # cleanup memory

            # retrieve structure features from structure space
            self.source.from_storage(filter={'structure_features':
                                             {'$exists': True}},
                                     projection={'material_id': 1,
                                                 'structure_features': 1,
                                                 '_id': 0})
            self.source.compress_memory(column='structure_features',
                                        decompress=True)
            self.source.memory.set_index('material_id', inplace=True)

            # save names of potential edges and their coordinates
            adjacent_verticies = self.source.memory.index.values
            sub_vectors = self.source.memory.loc[verticies].values
            all_vectors = self.source.memory.values
            self.source.memory = None  # cleanup memory

            # determine edge matrix
            edge_matrix = edge_calculator(all_vectors, sub_vectors, threshold)

            # store edges in graph space
            adjacency_list = {}
            for j in range(edge_matrix.shape[1]):
                adjacency_list[verticies[j]] = {
                    'edges': list(adjacent_verticies[edge_matrix[:, j]])}
            self.destination.memory = DataFrame.from_dict(
                adjacency_list, orient='index').reset_index().rename(
                    columns={'index': 'material_id'})
            self.destination.to_storage(identifier='material_id')

            return 1  # return True to continue the update

    def add_vacancy_edges(self, threshold=0.5, batch_size=10000,
                          edge_calculator=edg.sub_pairwise_squared_similarity,
                          featurizer=FRAMEWORK_FEATURIZER):
        '''
        add vacancy induced edges to the adjacency list

        Args:
            threshold (float) distance threshold to connect an edge
            batch_size (int) batch size for computing pairwise distances when
                generating graph edges. subject to memory constraints
            edge_calculator (func) a sub-pairwise distance calculator that
                returns an N x M adjacency matrix
        '''

        # retrieve structure features from structure space
        self.source.from_storage(filter={'structure_features':
                                         {'$exists': True}},
                                 projection={'material_id': 1,
                                             'structure_features': 1,
                                             'structure': 1,
                                             '_id': 0})
        self.source.compress_memory(column='structure_features',
                                    decompress=True)
        self.source.memory.set_index('material_id', inplace=True)

        # construct the numpy representation of the data
        ids = self.source.memory.index.values
        structures = self.source.memory['structure'].values
        self.source.memory.drop(labels='structure', axis=1, inplace=True)
        all_vectors = self.source.memory.values
        vector_labels = np.array(
            [s.split('.')[1] for s in self.source.memory.columns.values])
        self.source.memory = None

        # calculate edges for each possible vacancy structure
        for i, structure in enumerate(structures):

            # log calculated count
            print(i)

            # generate vacancy structure features
            base_structure = Structure.from_dict(structure)
            vacancies = [i.generate_defect_structure(supercell=(1, 1, 1))
                         for i in VacancyGenerator(base_structure)]
            sub_vectors = DataFrame(data={'structure': vacancies})
            featurizer.featurize_dataframe(sub_vectors, 'structure',
                                           ignore_errors=True, pbar=False)

            # establish same column order for edge calculation
            sub_vectors = sub_vectors[vector_labels]
            sub_vectors = sub_vectors.values
            edge_matrix = edge_calculator(all_vectors, sub_vectors, threshold)
            edge_mask = edge_matrix.sum(axis=1).astype(bool)
            vacancy_edges = set(ids[edge_mask])

            # update the adjacency list
            self.destination.from_storage(filter={'material_id': ids[i]},
                                          projection={'material_id': 1,
                                                      'edges': 1,
                                                      '_id': 0})
            prior_edges = set(self.destination.memory['edges'][0])
            self.destination.memory.at[
                self.destination.memory.index[0], 'edges'] =\
                list(prior_edges | vacancy_edges)
            self.destination.to_storage(identifier='material_id')

    # @in_batches
    # def calculate_distances(self, batch_size=10000,
    #                         dist_calculator=dist.sub_pairwise_squared_distance
    #                         ):
    #     '''
    #     solve for distances in the structure space. the graph distances are
    #     stored as an adjacency list using this mongodb document schema:
    #         'material_id' : str : the source vertex
    #         'distances' : dict : the destination verticies and their distances
    #                              {'mp_id': float}

    #     Args:
    #         batch_size (int) batch size for computing pairwise distances when
    #             generating graph edges. subject to memory constraints
    #         dist_calculator (func) a sub-pairwise distance calculator that
    #             returns an N x M adjacency matrix
    #     '''

    #     # load material ids without defined edges
    #     self.destination.from_storage(find={'filter':
    #                                         {'edges':
    #                                          {'$exists': False}},
    #                                         'projection':
    #                                         {'material_id': 1},
    #                                         'limit': batch_size})

    #     if len(self.destination.memory.index) == 0:

    #         return 0  # return False when update is complete

    #     else:

    #         # save names of verticies
    #         verticies = self.destination.memory['material_id'].values
    #         self.destination.memory = None  # cleanup memory

    #         # retrieve structure features from structure space
    #         self.source.from_storage(find={'filter':
    #                                        {'structure_features':
    #                                         {'$exists': True}},
    #                                        'projection':
    #                                        {'material_id': 1,
    #                                         'structure_features': 1,
    #                                         '_id': 0}})
    #         self.source.compress_memory(column='structure_features',
    #                                     decompress=True)
    #         self.source.memory.set_index('material_id', inplace=True)

    #         # save names of potential edges and their coordinates
    #         adjacent_verticies = self.source.memory.index.values
    #         sub_vectors = self.source.memory.loc[verticies].values
    #         all_vectors = self.source.memory.values
    #         self.source.memory = None  # cleanup memory

    #         # determine edge matrix
    #         edge_matrix = dist_calculator(all_vectors, sub_vectors)

    #         # store edges in graph space
    #         adjacency_list = {}
    #         for j in range(edge_matrix.shape[1]):
    #             adjacency_list[verticies[j]] = {
    #                 'edges': list(adjacent_verticies[edge_matrix[:, j]])}
    #         self.destination.memory = DataFrame.from_dict(
    #             adjacency_list, orient='index').reset_index().rename(
    #                 columns={'index': 'material_id'})
    #         self.destination.to_storage(identifier='material_id')

    #         return 1  # return True to continue the update


if __name__ == '__main__':

    gen = GenerateGraphCollection()
    gen.add_vacancy_edges()
