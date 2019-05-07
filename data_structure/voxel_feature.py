import common


class FeatureVoxel:

    def __init__(self, center):
        self.center = center
        self.feature_info_list = []

    def insert_feature(self, feature_info):
        self.feature_info_list.append(feature_info)


class FeatureInfo:

    def __init__(self, feature_list, distance):
        self.feature_list = feature_list
        self.distance = distance


class FeatureLidarPoint:

    def __init__(self, location, feature_list):
        self.location = location
        self.feature_list = feature_list
