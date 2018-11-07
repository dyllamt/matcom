from dataspace.base import Pipe, in_batches
from dataspace.workspaces.local_db import MongoFrame

from matcom.tools.edge_calculators import sub_pairwise_squared_distance

from pandas import DataFrame

'''
this module implements a pipeline for constructing graphs from feature spaces
'''


class ConstructStructureGraph(Pipe):
    '''
    structures (verticies) within a similarity threshold are connected by edges
    to form a graph of the structure space. additional edges connect structures
    that are similar to another when a defect (strain/vacancy/interstical) is
    introduced to one of the structures (e.g. rocksalt + intersticial = BCC)
    '''
    def __init__(self, structure_space, graph_space):
        '''
        Args:
            structure_space (MongoFrame) workspace with structure_features
            graph_space (MongoFrame) workspace holding verticies and edges
        '''
        Pipe.__init__(self, source=structure_space, destination=graph_space)

    def populate_verticies(self):
        '''
        populate verticies in graph space with verticies from structure space
        '''
        self.source.from_storage(find={'filter':
                                       {'structure_features':
                                        {'$exists': True}},
                                       'projection':
                                       {'material_id': 1}})
        self.transfer(to='destination')
        self.destination.to_storage(identifier='material_id', upsert=True)

    @in_batches
    def find_edges(self, threshold=0.5, batch_size=10000):
        '''
        solve for undirected edges connected to nodes in the structure space.
        jit compiling is essential for speed, since numpy broadcasting
        consumes too much memory for practical implementation

        Args:
            threshold (float) distance threshold to connect an edge
            batch_size (int) batch size for computing pairwise distances when
                generating graph edges. subject to memory constraints
        '''

        # load material ids without defined edges
        self.destination.from_storage(find={'filter':
                                            {'edges':
                                             {'$exists': False}},
                                            'projection':
                                            {'material_id': 1},
                                            'limit': batch_size})

        if len(self.destination.memory.index) == 0:

            return 0  # return False when update is complete

        else:

            # save names of verticies
            verticies = self.destination.memory['material_id'].values
            self.destination.memory = None  # cleanup memory

            # retrieve structure features from structure space
            self.source.from_storage(find={'filter':
                                           {'structure_features':
                                            {'$exists': True}},
                                           'projection':
                                           {'material_id': 1,
                                            'structure_features': 1,
                                            '_id': 0}})
            self.source.compress_memory(column='structure_features',
                                        decompress=True)
            self.source.memory.set_index('material_id', inplace=True)

            # save names of potential edges and their coordinates
            adjacent_verticies = self.source.memory.index.values
            sub_vectors = self.source.memory.loc[verticies].values
            all_vectors = self.source.memory.values
            self.source.memory = None  # cleanup memory

            # determine edge matrix
            edge_matrix = sub_pairwise_squared_distance(
                all_vectors, sub_vectors, threshold)

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

    def find_vacancy_edges(self):
        '''
        induce vacancy structures to find edges
        '''
        pass


if __name__ == '__main__':

    PATH = '/home/mdylla/repos/code/orbital_phase_diagrams/local_db'
    DATABASE = 'orbital_phase_diagrams'
    COLLECTION = 'structure'
    API_KEY = 'VerGNDXO3Wdt4cJb'

    structure_space = MongoFrame(PATH, DATABASE, COLLECTION)
    graph_space = MongoFrame(PATH, DATABASE, 'graph')

    pipe = ConstructStructureGraph(structure_space, graph_space)
    # pipe.populate_verticies()
    # print('copied verticies')
    # pipe.find_edges()
    # print('calculated edges')

    graph_space.from_storage(find={'projection': {'material_id': 1,
                                                  'edges': 1,
                                                  '_id': 0},
                                   'limit': 0})
    print(graph_space.memory)

    # data.from_storage(find={'filter':
    #                         {'structure_features': {'$exists': True}},
    #                         'projection':
    #                         {'material_id': 1,
    #                          'structure_features': 1,
    #                          '_id': 0}})
    # data.compress_memory(column='structure_features', decompress=True)
    # data.memory.set_index('material_id', inplace=True)

    # data.memory = data.memory.values

    # print('generating distances')

    # a = sub_pairwise_squared_distance(data.memory,
    #                                   data.memory[0:10000, :],
    #                                   0.7)

    # gen = GenerateMPStructureSpace(path=PATH, database=DATABASE,
    #                                collection=COLLECTION, api_key=API_KEY)
    # gen.featurize_structures()

    # space = AnalyzeAlloySpace(PATH, DATABASE, COLLECTION)

    # space.from_storage(find={'filter': {'material_id': 'mp-961652'},
    #                          'projection': {'structure': 1}})
    # parent = Structure.from_dict(space.memory['structure'][0])

    # # space.visulalize_alloy_space(parent, threshold=0.7)

    # space.from_storage(find={'filter': {'nsites': {'$lte': 5},
    #                                     'e_above_hull': {'$lte': 0.0}},
    #                          'projection': {'structure_features': 1,
    #                                         'material_id': 1,
    #                                         '_id': 0}})
    # space.compress_memory(column='structure_features', decompress=True)
    # space.memory.set_index('material_id', inplace=True)
    # space.memory.dropna(inplace=True)

    # scale = StandardScaler()
    # pca = PCA(n_components=2, whiten=True)
    # pipe = Pipeline([('scale', scale), ('pca', pca)])
    # components = pipe.fit_transform(space.memory)

    # traces = [go.Scatter(x=components[:, 0],
    #                      y=components[:, 1],
    #                      text=space.memory.index.values,
    #                      mode='markers')]
    # layout = go.Layout(hovermode='closest')
    # fig = go.Figure(data=traces, layout=layout)
    # py.plot(fig, filename='chrom_five')
    # print(space.memory)
