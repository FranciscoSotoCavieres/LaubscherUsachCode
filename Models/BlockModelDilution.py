import numpy as np
from Models.BlockModelStructure import BlockModelStructure
from Models.BlockModel import BlockModel
from Models.utils import polygon_area


class BlockModelDilution:
    _pde_percentage: float
    _pde_ratio: float
    _structure: BlockModelStructure
    dilution_coefficients: np.ndarray

    def __init__(self, pde_percentage: float, structure: BlockModelStructure):
        if pde_percentage <= 0 or pde_percentage > 100:
            raise Exception('Invalid PDE (%)')

        self._pde_percentage = pde_percentage
        self._pde_ratio = self._pde_percentage / 100
        self._structure = structure

    def get_pde_percentage(self):
        return self._pde_percentage

    def compute_dilution_coefficients(self):
        z_blocks: int = self._structure.shape[2]
        z_length: float = self._structure.block_size[2]
        x_length: float = self._structure.block_size[0]
        pde = self._pde_ratio

        # Create empty array for coefficients
        self.dilution_coefficients = np.zeros(
            [int(z_blocks / (pde / 3.0)), z_blocks])

        # Lines generation
        max_j: int = 0
        for i in range(1, z_blocks + 1):
            # Initial point
            x0 = 0
            y0 = pde * i * z_length

            # Middle point
            x1 = x_length / 2.0
            y1 = z_length * i
            slope = (y1 - y0) / (x1 - x0)

            # Calculate point
            x2 = x_length
            y2 = x_length * slope + y0

            initial_block, final_block = np.ceil(
                y0 / z_length), np.ceil(y2 / z_length)
            initial_x, initial_y = x0, y0
            count = 1
            for j in range(int(initial_block), int(final_block)):
                final_y = j * z_length
                final_x = (final_y - y0) / slope
                # Polygons
                if count == 1:  # Polygon of 5 sides
                    polygon_x = np.array([initial_x,
                                          final_x,
                                          x_length,
                                          x_length,
                                          0])
                    polygon_y = np.array([initial_y,
                                          final_y,
                                          j * z_length,
                                          (j - 1) * z_length,
                                          (j - 1) * z_length])
                    self.dilution_coefficients[j - 1, i -
                                               1] = polygon_area(polygon_x, polygon_y)
                else:  # Polygon of 4 sides
                    polygon_x = np.array([initial_x,
                                          final_x,
                                          x_length,
                                          x_length])
                    polygon_y = np.array([initial_y,
                                          final_y,
                                          j * z_length,
                                          (j - 1) * z_length])
                    self.dilution_coefficients[j - 1, i -
                                               1] = polygon_area(polygon_x, polygon_y)
                count += 1
                initial_x = final_x
                initial_y = final_y
            # Polygon of 3 sides
            polygon_x = np.array([initial_x, x2, x_length])
            polygon_y = np.array([initial_y, y2, initial_y])
            self.dilution_coefficients[int(
                final_block) - 1, i - 1] = polygon_area(polygon_x, polygon_y)

            if max_j < int(final_block) - 1:
                max_j = int(final_block)

        # Extract useful data
        self.dilution_coefficients = self.dilution_coefficients[:max_j + 1, :]

        # Areas treatment
        m, n = self.dilution_coefficients.shape  # Dimensions
        self.dilution_coefficients = np.insert(self.dilution_coefficients, n, values=0,
                                               axis=1)  # Add new columns

        start_independent_area = 0
        final_independent_area = n - 1
        for i in range(m):
            for k in range(n):
                if self.dilution_coefficients[i, k] != 0:
                    start_independent_area = k
                    break
            for k in range(n - 1, -1, -1):
                if self.dilution_coefficients[i, k] != 0:
                    final_independent_area = k
                    break
            for j in range(n):
                # Search for initial data
                if (j > start_independent_area) & (j < final_independent_area + 1):
                    self.dilution_coefficients[i, j] = self.dilution_coefficients[i, j] - np.sum(
                        self.dilution_coefficients[i, :j])

            # Areas balance
            start_independent_area = n - 1
            for k in range(n - 1, -1, -1):
                if self.dilution_coefficients[i, k] != 0:
                    start_independent_area = k
                    break
            self.dilution_coefficients[i, start_independent_area + 1] = x_length * z_length - np.sum(
                self.dilution_coefficients[i, :])

        self.dilution_coefficients = self.dilution_coefficients / \
            (x_length * z_length)

    def dilute_dataset(self, model: np.ndarray, topographyFraction: np.ndarray = None,
                       thinner: float = 0.0) -> np.ndarray:
        if model.shape[0] != self._structure.shape[0] \
                or model.shape[1] != self._structure.shape[1] \
                or model.shape[2] != self._structure.shape[2]:
            raise Exception('Dimensions mismatch')

        diluted_model = np.zeros(model.shape)  # Create an empty array
        if topographyFraction is None:
            topographyFraction = np.ones(
                [model.shape[0], model.shape[1]]) * model.shape[2] - 1  # Max topography
            topographyFraction = topographyFraction.astype('int')
        for i in range(model.shape[0]):
            for j in range(model.shape[1]):
                column = np.dot(
                    self.dilution_coefficients[:topographyFraction[i,
                                                                   j] + 1, :topographyFraction[i, j] + 1],
                    model[i, j, :topographyFraction[i, j] + 1])  # Dilute the column
                # Set the thinner
                column += self.dilution_coefficients[:
                                                     topographyFraction[i, j] + 1, -1] * thinner
                diluted_model[i, j, :topographyFraction[i, j] + 1] = column
        return diluted_model  # Returns diluted model

    def dilute_block_model(self, block_model: BlockModel, dataset_thinners: dict[str, float], topographyFraction: np.ndarray = None):
        structure = BlockModelStructure(
            block_model.structure.block_size, block_model.structure.shape, block_model.structure.offset)
        if (self.dilution_coefficients is None):
            raise ValueError(self.dilution_coefficients)
        diluted_block_model = BlockModel(structure)

        for key in dataset_thinners:
            thinner = dataset_thinners[key]
            original_data_set = block_model.get_data_set(key)
            diluted_data_set = self.dilute_dataset(
                original_data_set, topographyFraction, thinner)
            diluted_block_model.add_dataset(key, diluted_data_set)
        return diluted_block_model
