import matcom.tools.edge_calculators as edg

import numpy as np

from matcom.pipelines.generate_structure_collection import FRAMEWORK_FEATURIZER

from collections import defaultdict

from dataspace.base import Pipe, in_batches
from dataspace.workspaces.remote_db import MongoFrame

from pymatgen.core import Structure
from pymatgen.analysis.defects.generators import VacancyGenerator

from pandas import DataFrame

'''
this module implements a pipeline for generating a mongo database containing
a graph structure from a database of structural feature vectors. pipeline
operations are implemented as instance methods
'''


class GenerateGraphCollection(Pipe):
    '''
    structures (verticies) within a similarity threshold are connected by edges
    to form a graph of the structure space. additional edges connect structures
    that are similar to another when a defect (vacancy/interstical) is induced
    in one of the structures (Ex rocksalt + intersticial = BCC). the graph
    structure is stored as an adjacency list to conserve storage/memory.

    Notes: document schema for the graph collection
        "material_id" (str) the source vertex
        "edges" (list of str) the destination verticies
        "vacancy_edges" (dict) the destination verticies for each symmetrically
            inequivalant site (keys are site indicies, values are lists of str)

    Attributes:
        source (MongoFrame) a workspace which retrieves structural features
        destination (MongoFrame) a workspace which stores graph structure
    '''
    def __init__(self, host='localhost', port=27017,
                 database='structure_graphs',
                 structure_collection='structure', graph_collection='graph'):
        '''
        Args:
            host (str) hostname or IP address or Unix domain socket path
            port (int) port number on which to connect
            database (str) name of a pymongo database
            structure_collection (str) name of a pymongo collection that holds
                data on the structures being analyzed
            graph_collection (str) name of a pymongo collection that holds data
                on the graph representation of the structures
        '''
        structure_space = MongoFrame(
            host=host, port=port, database=database,
            collection=structure_collection)
        graph_space = MongoFrame(
            host=host, port=port, database=database,
            collection=graph_collection)
        Pipe.__init__(self, source=structure_space, destination=graph_space)

    def _load_structure_features(self):
        '''
        loads feature vectors into self.source.memory
        '''
        self.source.from_storage(filter={'structure_features':
                                         {'$exists': True}},
                                 projection={'material_id': 1,
                                             'structure_features': 1,
                                             '_id': 0})
        self.source.compress_memory(column='structure_features',
                                    decompress=True)
        self.source.memory.set_index('material_id', inplace=True)

    def _load_structures(self, material_ids):
        '''
        loads structures into self.source.memory
        '''
        self.source.from_storage(filter={'material_id':
                                         {'$in': material_ids}},
                                 projection={'material_id': 1,
                                             'structure': 1,
                                             '_id': 0})
        self.source.memory.set_index('material_id', inplace=True)
        self.source.memory = self.source.memory.loc[material_ids]

    def update_verticies(self,
                         criteria={'structure_features': {'$exists': True}}):
        '''
        populate verticies in graph space with verticies from structure space

        Notes:
            IO limited method
        '''
        self.source.from_storage(filter=criteria,
                                 projection={'material_id': 1})
        self.transfer(to='destination')
        self.destination.to_storage(identifier='material_id', upsert=True)

    @in_batches
    def update_edges(self, threshold=0.5, batch_size=10000,
                     edge_calculator=edg.pairwise_squared_similarity):
        '''
        solve for undirected, boolean edges based on exact similarity

        Notes:
            Tranformation limited method

        Args:
            threshold (float) distance threshold to connect an edge
            batch_size (int) batch size for computing pairwise distances when
                generating graph edges. subject to memory constraints
            edge_calculator (func) a pairwise edge calculator that returns an
                N x M adjacency matrix
        '''

        # load material ids without defined edges
        self.destination.from_storage(filter={'edges':
                                              {'$exists': False}},
                                      projection={'material_id': 1},
                                      limit=batch_size)

        if len(self.destination.memory.index) == 0:

            return 0  # returns False when update is complete

        else:

            # saves ids of source verticies from batch
            source_ids = self.destination.memory['material_id'].values
            self.destination.memory = None  # cleanup memory

            # saves the potential destination verticies and clean-up memory
            self._load_structure_features()
            all_ids = self.source.memory.index.values
            all_vectors = self.source.memory.values
            source_vectors = self.source.memory.loc[source_ids].values
            self.source.memory = None  # cleanup memory

            # determines edge matrix and coresponding adjacency list
            edge_matrix = edge_calculator(
                all_vectors, source_vectors, threshold)
            adjacency_list = {}
            for j in range(edge_matrix.shape[1]):
                adjacency_list[source_ids[j]] = {
                    'edges': list(all_ids[edge_matrix[:, j]])}

            # stores edges in the graph collection
            self.destination.memory = DataFrame.from_dict(
                adjacency_list, orient='index').reset_index().rename(
                    columns={'index': 'material_id'})
            self.destination.to_storage(identifier='material_id')

            return 1  # returns True to continue the update

    @in_batches
    def update_vacancy_edges(self, threshold=0.5, batch_size=100,
                             edge_calculator=edg.pairwise_squared_similarity,
                             featurizer=FRAMEWORK_FEATURIZER):
        '''
        solve for directed, boolean edges based on similarity with a vacancy

        Notes:
            Transformation limited method (featurization of vacancy structures)

        Args:
            threshold (float) distance threshold to connect an edge
            batch_size (int) batch size for computing pairwise distances when
                generating graph edges. subject to memory constraints
            edge_calculator (func) a sub-pairwise distance calculator that
                returns an N x M adjacency matrix
            featurizer (BaseFeaturizer) an instance of a structural featurizer
        '''

        # loads a batch of verticies without defined edges
        self.destination.from_storage(
            filter={'vacancy_edges':
                    {'$exists': False}},
            projection={'material_id': 1},
            limit=batch_size)

        if len(self.destination.memory.index) == 0:

            return 0  # returns False when update is complete

        else:

            # gets the source vertex ids for the current batch
            source_ids = self.destination.memory['material_id'].values
            self.destination.memory = None  # cleanup memory

            # gets the potential destination vertex ids and their features
            self._load_structure_features()
            all_ids = self.source.memory.index.values
            all_vectors = self.source.memory.values
            vector_labels = np.array(
                [s.split('.')[1] for s in self.source.memory.columns.values])
            self.source.memory = None  # cleanup memory

            # calculates feature vectors for each (source) vacancy structure
            self._load_structures(list(source_ids))
            source_structures = self.source.memory['structure'].values
            self.source.memory = None  # cleanup memory

            vacancy_structures = []
            for material_id, structure in zip(source_ids, source_structures):
                structure = Structure.from_dict(structure)
                for site_i, vacancy in enumerate(VacancyGenerator(structure)):
                    vacancies = [
                        material_id,
                        str(site_i),
                        vacancy.generate_defect_structure(supercell=(1, 1, 1))
                    ]
                    vacancy_structures.append(vacancies)
            vacancy_structures = DataFrame(
                data=vacancy_structures,
                columns=['source_id', 'site_index', 'structure'])

            vacancy_vectors = featurizer.featurize_dataframe(
                vacancy_structures, 'structure', ignore_errors=True,
                pbar=False, inplace=False)[vector_labels].values

            # determine edge matrix and coresponding adjacency list
            edge_matrix = edge_calculator(
                all_vectors, vacancy_vectors, threshold)
            adjacency_list = defaultdict(dict)
            for j in range(edge_matrix.shape[1]):
                source_id = vacancy_structures['source_id'][j]
                site_index = vacancy_structures['site_index'][j]
                adjacency_list[source_id][site_index] = list(
                    all_ids[edge_matrix[:, j]])

            # store edges in graph space
            self.destination.memory = DataFrame.from_records(
                list(adjacency_list.items()),
                columns=['material_id', 'vacancy_edges'])
            self.destination.to_storage(identifier='material_id')

            return 1  # return True to continue the update


if __name__ == '__main__':

    gen = GenerateGraphCollection()
    # gen.destination.delete_storage(clear_collection=True)
    # gen.update_verticies()
    # gen.update_edges()
    # gen.update_vacancy_edges()
