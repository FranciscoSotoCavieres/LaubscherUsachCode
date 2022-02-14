import numpy as np
from openpyxl import Workbook
from Models.BlockModelStructure import BlockModelStructure
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

            if self.footprint_indices[self.footprint_indices < 0].size > 0 or \
                    self.footprint_indices[self.footprint_indices > z_blocks].size > 0:
                raise Exception('Invalid footprint indices')
        except Exception as e:
            raise e
        self.footprint_height = footprint_indices * z_size

    def export_to_excel(self, filepath):
        workbook = Workbook()

        export_matrix(self.footprint_indices, workbook, index_keyword, 0)
        export_matrix(self.footprint_height, workbook, height_keyword, 0)
        remove_default_worksheet(workbook)

        workbook.save(filepath)
