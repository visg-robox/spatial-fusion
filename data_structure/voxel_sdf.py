import common


class Voxel:

    def __init__(self, center, vertex_list=[]):
        self.center = self.location_to_center(center)
        self.vertex_list = vertex_list


class Vertex:

    def __init__(self, sdf, location):
        self.sdf = sdf
        self.location = location

    @staticmethod
    def location_to_key(location):
        location[0] = round(location[0]/common.voxel_length)
        location[1] = round(location[1]/common.voxel_length)
        location[2] = round(location[2]/common.voxel_length)
        return tuple(location)

