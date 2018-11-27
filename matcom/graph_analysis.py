from dataspace.workspaces.local_db import MongoFrame
from dataspace.workspaces.local_db import local_connection

from graph_tool.all import Graph, load_graph, remove_parallel_edges,\
    remove_self_loops, sfdp_layout, graph_draw

'''
this module implements a workspace for analyzing a local graph structure
'''


class GraphAnalysis(MongoFrame):
    '''
    graph analyis is uses tools from the graph-tool library
    '''
    def __init__(self, path, database, collection):
        '''
        Args:
            path (str) path to a mongodb directory
            database (str) name of a pymongo database
            collection (str) name of a pymongo collection
        '''
        MongoFrame.__init__(self, path, database, collection)

    def gen_graph_from_mongo(self):
        '''
        load graph structure from storage. note that add_edge_list will not
        match vertex ids (str ids) in subsequent calls of the function
        '''

        self.from_storage(find={'projection': {'material_id': 1,
                                               'edges': 1}})
        sources = self.memory['material_id']
        destinations = self.memory['edges']

        self.memory = None  # cleanup memory attribute

        print('loaded data structures')

        edge_list = [(sources[i], destinations[i][j]) for i in
                     range(len(sources)) for j in range(len(destinations[i]))]

        print('generated edge list')

        sources = None  # cleanup temporary data variables
        destinations = None

        graph = Graph(directed=False)
        graph.add_edge_list(edge_list, hashed=True, string_vals=True)

        return graph

    def gen_sub_graph_from_mongo(self, center, snn=1):
        '''
        load graph structure from storage. note that add_edge_list will not
        match vertex ids (str ids) in subsequent calls of the function

        Args:
            center (str) mp-id of the center of the graph
            snn (int) the number of second nearest neighbors to expand to
        '''

        edge_list = []

        self.from_storage(find={'filter': {'material_id': center},
                                'projection': {'material_id': 1,
                                               'edges': 1}})
        sources = self.memory['material_id'][0]
        destinations = self.memory['edges'][0]

        edge_list.extend([(sources, destinations[j]) for j in
                          range(len(destinations))])

        for i in range(snn):
            self.from_storage(find={'filter': {'material_id':
                                               {'$in': destinations}},
                                    'projection': {'material_id': 1,
                                                   'edges': 1}})
            sources = self.memory['material_id']
            destinations = self.memory['edges']

            edge_list.extend([(sources[i], destinations[i][j]) for i in
                              range(len(sources)) for j in
                              range(len(destinations[i]))])

            destinations = [destinations[i][j] for i in
                            range(len(sources)) for j in
                            range(len(destinations[i]))]

        print('generated edge list')

        graph = Graph(directed=False)
        graph.add_edge_list(edge_list, hashed=True, string_vals=True)

        return graph


if __name__ == '__main__':

    PATH = '/home/mdylla/repos/code/orbital_phase_diagrams/local_db'
    DATABASE = 'orbital_phase_diagrams'
    COLLECTION = 'graph'

    # graph_space = GraphAnalysis(PATH, DATABASE, COLLECTION)

    # graph = graph_space.gen_sub_graph_from_mongo(center='mp-961652')
    # remove_parallel_edges(graph)
    # remove_self_loops(graph)
    graph = load_graph('hh_network.gt', fmt='gt')
    # graph.save('hh_network.gt', fmt='gt')
    pos = sfdp_layout(graph)
    graph_draw(graph, pos=pos, output="hh-sfdp.png")

    # g = load_graph('pruned_structure_network.gt', fmt='gt')
    # pos = sfdp_layout(g)
    # graph_draw(g, pos=pos, output="graph-draw-sfdp.pdf")
    # remove_parallel_edges(g)
    # remove_self_loops(g)
    # g.save('pruned_structure_network.gt', fmt='gt')
    # print(g)

    # print(g)
