from dataspace.workspaces.local_db import MongoFrame
from dataspace.workspaces.local_db import local_connection

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

    @local_connection
    def load_graph_structure(self):
        '''
        load graph structure from storage. the data is not loaded with the
        builtin from_memory() method to avoid excessive use of memory
        '''

        cursor = self.connection.find()
        for document in cursor:
            vertex = document['material_id']
            connections = document['edges']

            
