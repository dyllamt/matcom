from dataspace.workspaces.local_db import MongoFrame

from pandas import DataFrame, concat

from pymatgen.analysis.defects.generators import VacancyGenerator

import numpy as np


class AnalyzeAlloySpace(MongoFrame):
    '''
    a substitutional alloy space contains structures with the same sites or
    structures related by either a single vacancy or intersticial defect. a
    structure prototype is used to expand an alloy space for a material family.
    similar structures to the prototype and its defect structures (exemplars)
    are found using a distance metric of structural feature vectors, which are
    stored in a local database of structures. the neighborhoods of compounds
    that are similar to the exemplars can be analyzed using heirarchical
    principle component analysis, which visualizes the maximum variance in and
    among the neighborhoods of structures.

    local structure database must contain "material_id", "structure", and
    "structure_features" fields. MPStructureSpace constructs a compatible
    database for this workspace, but could be extended to the OQMD structures

    Attributes:

        alloy attributes
        . parent (Structure) pymatgen object that alloy space is built around
        . exemplars (DataFrame) contains both pymatgen Structures and their
            feature vectors. an exemplar (row of DataFrame) is a structure that
            is either the parent structure or a vacancy/intersticial structure.
            the first row is the exemplar of the parent structure
        . distances (DataFrame) pairwise distances matrix of the database
            structures (row) and exemplars (column)
        . neighborhoods ([DataFrame]) the index contains the unique database
            entry identifier, the data contains the feature vectors of
            each structure in the neighborhood of each exemplar (list order is
            preserved from row order of exemplars attribute)

        inherited from MongoFrame:
        . path (str) path to local mongodb
        . database (str) name of the database
        . collection (str) name of the collection
        . connection (Collection|None) statefull pymongo connection to storage
        . memory (DataFrame|None) temporary data storage
    '''

    def __init__(self, path, database, collection):
        '''
        Args:
            path (str) path to a mongodb directory
            database (str) name of a pymongo database
            collection (str) name of a pymongo collection
        '''
        MongoFrame.__init__(self, path, database, collection)
        self.parent = None
        self.exemplars = None
        self.distances = None
        self.neighborhoods = None

    def visulalize_alloy_space(self, parent, threshold=0.1):
        '''
        perform principle component analysis on structures that are within the
        neighborhood of the parent and defect structures set by threshold
        '''
        self.parent = parent
        self.exemplars = self.generate_exemplars()
        self.distances = self.generate_distances()
        self.neighborhoods = self.generate_neighborhoods(threshold)
        self.memory = None  # clear large ammount of data in temporary storage

        combined_neighborhoods = concat(self.neighborhoods, axis=0)
        pca = PCA(n_components=2, whiten=True)
        pca.fit(combined_neighborhoods)

        traces = []
        for neighborhood in self.neighborhoods:
            components = pca.transform(neighborhood)
            traces.append(go.Scatter(x=components[:, 0],
                                     y=components[:, 1],
                                     text=neighborhood.index.values,
                                     mode='markers'))
        fig = go.Figure(data=traces)
        py.plot(fig, filename='test_figure')

    def generate_exemplars(self):
        '''
        generate feature vectors of parent compound and its defect structures

        Returns (DataFrame) exemplar structures and their feature vectors. the
            first row is the parent, the remaining are defect structures
        '''

        # generate exemplar structures
        exemplars = [self.parent]
        vgen = VacancyGenerator(structure=self.parent)
        exemplars.extend([i.generate_defect_structure(supercell=(1, 1, 1))
                          for i in vgen])

        # featurize exemplar structures
        featurizer = SiteStatsFingerprint.from_preset(
            preset='CrystalNNFingerprint_ops',
            stats=['mean', 'std_dev', 'maximum', 'minimum'])
        exemplars = DataFrame.from_dict({'structure': exemplars})
        featurizer.featurize_dataframe(exemplars, 'structure',
                                       ignore_errors=False, pbar=False)
        return exemplars

    def generate_distances(self):
        '''
        generate pairwise distances between database structures and exemplars

        Returns (DataFrame) distances between database structures (rows) and
            and the instance exemplars (columns)
        '''

        # load structural features from database of structures
        self.from_storage(find={'filter':
                                {'structure_features': {'$exists': True}},
                                'projection':
                                {'material_id': 1,
                                 'structure_features': 1,
                                 '_id': 0}
                                })
        self.compress_memory(column='structure_features', decompress=True)
        self.memory.set_index('material_id', inplace=True)
        self.memory.sort_index(axis=1, inplace=True)
        database_features = self.memory.values

        # compute pairwise distances with numpy broadcasting
        exemplar_features = self.exemplars.drop(
            columns=['structure']).sort_index(axis=1).values
        distances = np.sqrt(
            np.sum((database_features[:, np.newaxis, :] -
                    exemplar_features[np.newaxis, :, :]) ** 2., axis=2))

        # return result as a dataframe
        return DataFrame(data=distances, index=self.memory.index.values,
                         columns=self.exemplars.index.values.astype(str))

    def generate_neighborhoods(self, threshold):
        '''
        generate neighborhoods around each exemplar within threshold distance.
        the data in self.memory is reused from the generate_distances()
        function call durring instance initiation

        Returns ([DataFrame]) a list of neighborhoods. each neighborhood
            contains the mp ids (index) and the feature vectors (data)
        '''

        distances = self.distances.values

        neighborhoods = []
        for j in range(distances.shape[1]):
            distances_from_exemplar_j = distances[:, j]
            bool_within_threshold = distances_from_exemplar_j <= threshold
            neighborhoods.append(
                self.memory.loc[bool_within_threshold])
        return neighborhoods
