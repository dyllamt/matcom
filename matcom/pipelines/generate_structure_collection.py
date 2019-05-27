from dataspace.base import Pipe, in_batches
from dataspace.workspaces.materials_api import APIFrame
from dataspace.workspaces.remote_db import MongoFrame

from pandas import concat

from pymatgen.core.structure import Structure

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint

'''
this module implements a pipeline for generating a mongo database of structures
and their feature vectors from the materials project database. pipeline
operations are implemented as instance methods
'''


# transformer for converting structures into feature vectors
FRAMEWORK_FEATURIZER = SiteStatsFingerprint(
    site_featurizer=CrystalNNFingerprint.from_preset(
        preset="ops", distance_cutoffs=None, x_diff_weight=0.0,
        porous_adjustment=False),
    stats=['mean', 'std_dev', 'maximum', 'minimum'])


class GenerateStructureCollection(Pipe):
    '''
    structures are collected from the materials project and deposited in a
    mongodb collection under the document field "structure". these structures
    are featurized using transformers from the matminer library. the feature
    vectors are stored as dictionaries in the "structure_features" field of the
    collection. the data is indexed by the document field "material_id"

    Notes: document schema for the graph collection
        material_id (str) the unique identifier for a material
        structure (dict) representation of a pymatgen Structure
        structure_features (dict) features describing the local coordination
            enviornments in a given structure

    Attributes:
        source (APIFrame) a workspace which retrieves materials project data
        destination (MongoFrame) a workspace which stores structure data
    '''
    def __init__(self, host='localhost', port=27017,
                 database='structure_graphs', collection='structure',
                 api_key=None):
        '''
        Args:
            host (str) hostname or IP address or Unix domain socket path
            port (int) port number on which to connect
            database (str) storage database
            collection (str) storage collection
            api_key (str) materials project api key
        '''
        mp_retriever = APIFrame(
            RetrievalSubClass=MPDataRetrieval, api_key=api_key)
        structure_space = MongoFrame(
            host=host, port=port, database=database, collection=collection)
        Pipe.__init__(self, source=mp_retriever, destination=structure_space)

    def update_material_ids(self, criteria={'structure': {'$exists': True},
                                            'nsites': {'$lte': 50}},
                            initial_collection=False):
        '''
        inserts documents in the mongodb collection based on a filter posted
        against the materials project database

        Notes:
            IO limited method

        Args:
            criteria (son) a pymongo-like filter to match against entries in
                the materials project database. if an entry is a match, then
                its corresponding "material_id" is retrieved from the database
            initial_collection (bool) set to true if this is the initial
                collection of ids (i.e. will perform non-unique insertion)

        DB operation:
            upserts "material_id" fields on documents by "material_id"
        '''
        self.source.from_storage(criteria=criteria,
                                 properties=['material_id'],
                                 index_mpid=False)
        self.transfer(to='destination')

        if initial_collection:  # performs non-unique insertion
            identifier = None
        else:  # will check if material_id is already in collection
            identifier = 'material_id'
        self.destination.to_storage(identifier=identifier, upsert=True)

    @in_batches
    def update_structures(self, batch_size=500):
        '''
        collects structures for all documents missing a "structure" field

        Notes:
            IO limited method

        Args:
            batch_size (int) limit on number of structures retrieved at a time

        DB operation:
            upserts "structure", "formation_energy_per_atom", "e_above_hull",
            and "nsites" fields on documents by "material_id"
        '''

        # load materials without structures from storage
        self.destination.from_storage(
            filter={'structure': {'$exists': False}},
            projection={'material_id': 1,
                        '_id': 0},
            limit=batch_size)

        if len(self.destination.memory.index) == 0:

            return 0  # return False when update is complete

        else:

            # retrieve materials data from mp database
            material_ids = list(self.destination.memory['material_id'])
            self.source.from_storage(
                criteria={'material_id': {'$in': material_ids}},
                properties=['material_id', 'structure',
                            'formation_energy_per_atom', 'e_above_hull',
                            'nsites'],
                index_mpid=False)

            # converts pymatgen Structure objects into dictionaries
            self.source.memory['structure'] = [
                i.as_dict() for i in self.source.memory['structure']]

            # save materials data to local storage
            self.transfer(to='destination')
            self.destination.to_storage(identifier='material_id', upsert=True)

            return 1  # return True to continue update

    @in_batches
    def update_structure_features(self, batch_size=500,
                                  featurizer=FRAMEWORK_FEATURIZER):
        '''
        computes feature vectors for all documents missing the
        "structure_features" document field

        Notes:
            Transformation limited method (featurization step)

        Args:
            batch_size (int) limit on number of structures featurized at a time
            featurizer (BaseFeaturizer) an instance of a structural featurizer

        DB operation:
            upserts "structure_features" fields on documents by "material_id"
        '''

        # load structures that are not featurized from storage
        self.destination.from_storage(
            filter={'structure': {'$exists': True},
                    'structure_features': {'$exists': False}},
            projection={'material_id': 1,
                        'structure': 1,
                        '_id': 0},
            limit=batch_size)

        if len(self.destination.memory.index) == 0:

            return 0  # return False when update is complete

        else:

            # featurize loaded structures
            self.destination.memory['structure'] =\
                [Structure.from_dict(struct) for struct in
                    self.destination.memory['structure']]
            featurizer.featurize_dataframe(self.destination.memory,
                                           'structure',
                                           ignore_errors=True,
                                           pbar=False, inplace=True)

            # store compressed features in permanant storage
            mp_ids = self.destination.memory[['material_id']]
            self.destination.memory.drop(columns=['material_id', 'structure'],
                                         inplace=True)
            self.destination.compress_memory('structure_features')
            self.destination.memory = concat([mp_ids, self.destination.memory],
                                             axis=1)
            self.destination.to_storage(identifier='material_id', upsert=True)

            return 1  # return True to continue update


if __name__ == '__main__':

    gen = GenerateStructureCollection()
    # gen.destination.delete_storage(clear_collection=True)
    # gen.update_material_ids(initial_collection=True)
    # gen.update_structures()
    # gen.update_structure_features()
