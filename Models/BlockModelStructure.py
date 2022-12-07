import numpy as np
from numpy import ndarray


class BlockModelStructure:
    block_size: ndarray
    offset: ndarray
    shape: np.ndarray

    def __init__(self, block_size: np.ndarray, shape: np.ndarray, offset: np.ndarray):
        self.offset = offset
        self.shape = shape
        self.block_size = block_size

    def get_block_volume(self) -> float:
        return self.block_size[0] * self.block_size[1] * self.block_size[2]

    def get_block_area(self) -> float:
        return self.block_size[0] * self.block_size[1]

    def check_dimensions(self, data: np.ndarray):
        if len(data.shape) != 3:
            return False
        if data.shape[0] == self.shape[0] and data.shape[1] == self.shape[1] and data.shape[2] == self.shape[2]:
            return True
        return False

    def get_subscripts(self, x: float, y: float, z: float):
        i = self.get_subscript(x, 'i')
        j = self.get_subscript(y, 'j')
        k = self.get_subscript(z, 'k')
        return np.array([i, j, k], dtype=int)

    def get_centroid(self, subscript_i: int, subscript_j: int, subscript_k: int):
        x = subscript_i * self.block_size[0] + self.offset[0]
        y = subscript_j * self.block_size[1] + self.offset[1]
        z = subscript_k * self.block_size[2] + self.offset[2]

        return np.array([x, y, z])

    def get_left_down_near_corner(self):
        x = - self.block_size[0] / 2 + self.offset[0]
        y = - self.block_size[1] / 2 + self.offset[1]
        z = - self.block_size[2] / 2 + self.offset[2]
        return np.array([x, y, z])

    def get_subscript(self, value: float, subscript_type='i') -> int:
        index = 0
        if subscript_type == 'j':
            index = 1
        elif subscript_type == 'k':
            index = 2

        init = self.offset[index]
        block_size = self.block_size[index]
        size = self.shape[index]

        subscript = (value - init) / block_size
        if subscript >= size:
            return int(subscript - 1)
        if subscript < 0:
            return 0
        return int(subscript)

    def get_data_set_from_1D(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, datasets: dict[str, np.ndarray]):

        indices = np.zeros([x.shape[0], 3], dtype=int)
        for i in range(x.shape[0]):
            indices[i, :] = self.get_subscripts(x[i], y[i], z[i])

        output: dict[str, np.ndarray] = dict([])
        for dataset_name in datasets.keys():
            data_1d = datasets[dataset_name]
            array = np.zeros(self.shape)
            for i in range(x.shape[0]):
                array[indices[i, 0], indices[i, 1], indices[i, 2]] = data_1d[i]
            output[dataset_name] = array
        return output

    def clam_subscripts(self, i: int, j: int, k: int):
        if (i >= self.shape[0]):
            i = self.shape[0] - 1
        elif (i < 0):
            i = 0

        if (j >= self.shape[1]):
            j = self.shape[1] - 1
        elif (j < 0):
            j = 0

        if (k >= self.shape[2]):
            k = self.shape[2] - 1
        elif (k < 0):
            k = 0

        return (i, j, k)

    @staticmethod
    def from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray):
        unique_x: np.ndarray = np.unique(x)
        unique_y: np.ndarray = np.unique(y)
        unique_z: np.ndarray = np.unique(z)

        unique_x.sort()
        unique_y.sort()
        unique_z.sort()

        block_size_x: float = unique_x[1] - unique_x[0]
        block_size_y: float = unique_y[1] - unique_y[0]
        block_size_z: float = unique_z[1] - unique_z[0]

        offset_x: float = unique_x[0]
        offset_y: float = unique_y[0]
        offset_z: float = unique_z[0]

        shape_x: int = unique_x.shape[0]
        shape_y: int = unique_y.shape[0]
        shape_z: int = unique_z.shape[0]

        return BlockModelStructure(np.array([block_size_x, block_size_y, block_size_z]),
                                   np.array([shape_x, shape_y, shape_z]),
                                   np.array([offset_x, offset_y, offset_z]))
