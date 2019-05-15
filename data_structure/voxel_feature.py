import common


class FeatureVoxel:

    def __init__(self, center):
        self.center = center
        self.feature_info_list = []

    def insert_feature(self, feature_info):
        self.feature_info_list.append(feature_info)


class FeatureInfo:

    def __init__(self, feature_list, vector, near_keys):
        self.feature_list = feature_list
        self.vector = vector

class FeatureInfo_new:

    def __init__(self, feature_list, vector, near_keys):
        self.feature_list = feature_list
        self.vector = vector
        self.near_keys = near_keys

class FeatureLidarPoint:

    def __init__(self, location, feature_list):
        self.location = location
        self.feature_list = feature_list
