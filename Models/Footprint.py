import numpy as np
from openpyxl import Workbook
from Models.BlockModelStructure import BlockModelStructure
from Models.BlockModel import BlockModel
from Models.excel_utils import export_matrix, remove_default_worksheet


index_keyword = 'Index'
height_keyword = 'Height'


class Footprint:
    structure: BlockModelStructure
    footprint_indices: np.ndarray
    footprint_height: np.ndarray

    def __init__(self, footprint_indices: np.ndarray, structure: BlockModelStructure):
        self.footprint_indices = footprint_indices.astype('int')
        self.structure = structure

        x_blocks = self.structure.shape[0]
        y_blocks = self.structure.shape[1]
        z_blocks = self.structure.shape[2]
        z_size = self.structure.block_size[2]

        # Check dimensions
        try:
            # Check dimensions
            if self.footprint_indices.shape[0] != x_blocks or self.footprint_indices.shape[1] != y_blocks:
                raise Exception(
                    'Block model and footprint dimensions doesn''t match')

            # if self.footprint_indices[self.footprint_indices < 0].size > 0 or \
            #         self.footprint_indices[self.footprint_indices > z_blocks].size > 0:
            #     raise Exception('Invalid footprint indices')
        except Exception as e:
            raise e
        self.footprint_height = footprint_indices * z_size

    def export_to_excel(self, filepath):
        workbook = Workbook()

        export_matrix(self.footprint_indices, workbook, index_keyword, 0)
        export_matrix(self.footprint_height, workbook, height_keyword, 0)
        remove_default_worksheet(workbook)

        workbook.save(filepath)

    def compute_slope_percentage(self):

        size_x = self.structure.block_size[0]
        size_y = self.structure.block_size[1]

        slope_percentage_matrix = np.zeros(
            [self.structure.shape[0], self.structure.shape[1]])

        for i in range(self.footprint_height.shape[0]):
            for j in range(self.footprint_height.shape[1]):

                (im1, _, _) = self.structure.clamp_subscripts(i-1, 0, 0)
                (i0, _, _) = self.structure.clamp_subscripts(i, 0, 0)
                (ip1, _, _) = self.structure.clamp_subscripts(i+1, 0, 0)

                (_, jm1, _) = self.structure.clamp_subscripts(0, j-1, 0)
                (_, j0, _) = self.structure.clamp_subscripts(0, j, 0)
                (_, jp1, _) = self.structure.clamp_subscripts(0, j+1, 0)

                zIm1Jp1 = self.footprint_height[im1, jp1]
                zI0Jp1 = self.footprint_height[i0, jp1]
                zIp1Jp1 = self.footprint_height[ip1, jp1]

                zIm1Jm1 = self.footprint_height[im1, jm1]
                zI0Jm1 = self.footprint_height[im1, jm1]
                zIp1Jm1 = self.footprint_height[im1, jm1]

                zIm1J0 = self.footprint_height[im1, j0]
                zI0J0 = self.footprint_height[i0, j0]
                zIp1J0 = self.footprint_height[im1, j0]

                if (zIm1Jp1 == 0):
                    zIm1Jp1 = zI0J0
                if (zI0Jp1 == 0):
                    zI0Jp1 = zI0J0
                if (zIp1Jp1 == 0):
                    zIp1Jp1 = zI0J0

                if (zIm1Jm1 == 0):
                    zIm1Jm1 = zI0J0
                if (zI0Jm1 == 0):
                    zI0Jm1 = zI0J0
                if (zIp1Jm1 == 0):
                    zIp1Jm1 = zI0J0

                if (zIm1J0 == 0):
                    zIm1J0 = zI0J0
                if (zIp1J0 == 0):
                    zIp1J0 = zI0J0

                ew = ((zIp1Jp1 + 2 * zIp1J0 + zIp1Jm1) - (zIm1Jp1 + 2 * zIm1J0 + zIm1Jm1)) / (8 * size_x)
                ns = ((zIp1Jp1 + 2 * zI0Jp1 + zIm1Jp1) - (zIp1Jm1 + 2 * zI0Jm1 + zIm1Jm1)) / (8 * size_y)
                slope_percentage = 100 * ((ew ** 2)+(ns ** 2)) ** 0.5
                slope_percentage_matrix[i, j] = slope_percentage
        return slope_percentage_matrix
    

        
