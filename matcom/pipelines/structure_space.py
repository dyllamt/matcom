from dataspace.base import Pipe, in_batches
from dataspace.workspaces.materials_api import MPFrame
from dataspace.workspaces.local_db import MongoFrame

from pandas import concat

from pymatgen.core.structure import Structure

from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint

'''
this module implements a pipeline for generating structural feature vectors
'''


FRAMEWORK_FEATURIZER = SiteStatsFingerprint(
    site_featurizer=CrystalNNFingerprint.from_preset(
        preset="ops", distance_cutoffs=None, x_diff_weight=0.0,
        porous_adjustment=False),
    stats=['mean', 'std_dev', 'maximum', 'minimum'])


class GenerateStructureSpace(Pipe):
    '''
    structure data is collected from the materials project and deposited in a
    local mongodb. structures in the local db are featurized with matminer.
    feature vectors are stored as dictionaries in the structure_features field
    '''
    def __init__(self, path, database, collection, api_key=None):
        '''
        Args:
            path (str) path to a local mongodb directory
            database (str) name of a pymongo database
            collection (str) name of a pymongo collection
            api_key (str) materials project api key
        '''
        self.source = MPFrame(api_key=api_key)
        self.destination = MongoFrame(path=path, database=database,
                                      collection=collection)

    def update_all(self, batch_size=500, featurizer=FRAMEWORK_FEATURIZER):
        '''
        convienience function for updating the structure space
        '''
        self.update_mp_ids()
        self.update_structures(batchsize=batch_size)
        self.featurize_structures(batch_size=batch_size, featurizer=featurizer)

    def update_mp_ids(self):
        '''
        update the collection of mp ids in the local database
        '''
        self.source.from_storage(criteria={'structure': {'$exists': True}},
                                 properties=['material_id'],
                                 index_mpid=False)
        self.transfer(to='destination')
        self.destination.to_storage(identifier='material_id', upsert=True)

    @in_batches
    def update_structures(self, batch_size=500):
        '''
        update the collection of structures in the local database. updates are
        done in batches to accomadate the api limits of the materials project

        Args:
            batch_size (int) limit on number of structures retrieved at a time
        '''

        # load material ids without structure to memory
        self.destination.from_storage(find={'filter':
                                            {'structure':
                                             {'$exists': False}},
                                            'projection':
                                            {'material_id': 1},
                                            'limit': batch_size})

        if len(self.destination.memory.index) == 0:

            return 0  # return False when update is complete

        else:

            # retrieve materials data from mp database
            mp_ids = list(self.destination.memory['material_id'].values)
            self.source.from_storage(
                criteria={'material_id': {'$in': mp_ids}},
                properties=['material_id', 'structure',
                            'formation_energy_per_atom', 'e_above_hull',
                            'nsites'],
                index_mpid=False)

            # save materials data to local storage
            self.transfer(to='destination')
            self.destination.to_storage(identifier='material_id', upsert=True)

            return 1  # return True to continue update

    @in_batches
    def featurize_structures(self, batch_size=500,
                             featurizer=FRAMEWORK_FEATURIZER):
        '''
        featurize structures in the local database in batches

        Args:
            batch_size (int) limit on number of structures retrieved at a time
            featurizer (BaseFeaturizer) an instance of a structural featurizer
        '''

        # load structures that are not featurized from storage
        self.destination.from_storage(
            find={'filter': {'structure': {'$exists': True},
                  'structure_features': {'$exists': False}},
                  'projection': {'material_id': 1,
                                 'structure': 1,
                                 '_id': 0},
                  'limit': batch_size})

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
                                           pbar=False)

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
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # 0. user fields
    # PATH = '/home/mdylla/repos/code/orbital_phase_diagrams/local_db'
    # DATABASE = 'orbital_phase_diagrams'
    # COLLECTION = 'structure'
    # API_KEY = 'VerGNDXO3Wdt4cJb'

    # 1. generate structure space
    # gen = GenerateStructureSpace(PATH, DATABASE, COLLECTION, API_KEY)
    # gen.update_all()

    # 2. visualize structure space
    # data = MongoFrame(PATH, DATABASE, COLLECTION)
    # data.from_storage(find={'filter':
    #                         {'structure_features': {'$exists': True}},
    #                         'projection':
    #                         {'material_id': 1,
    #                          'structure_features': 1,
    #                          '_id': 0}})
    # data.compress_memory(column='structure_features', decompress=True)
    # data.memory.set_index('material_id', inplace=True)