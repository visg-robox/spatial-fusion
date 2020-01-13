

class SemanticVoxel:

    def __init__(self, center):
        self.center = center
        self.semantic_info_list = []

    def insert_label(self, semantic_info):
        self.semantic_info_list.append(semantic_info)


class SemanticInfo:

    def __init__(self, label_list, geometry_feature=None):
        self.label_list = label_list
        self.geometry_feature = geometry_feature


class SemanticLidarPoint:

    def __init__(self, location, label_list):
        self.location = location
        self.label_list = label_list

