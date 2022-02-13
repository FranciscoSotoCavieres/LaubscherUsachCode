import numpy as np
from openpyxl import Workbook
from Models.BlockModelStructure import BlockModelStructure

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
                raise Exception('Block model and footprint dimensions doesn''t match')

            if self.footprint_indices[self.footprint_indices < 0].size > 0 or \
                    self.footprint_indices[self.footprint_indices > z_blocks].size > 0:
                raise Exception('Invalid footprint indices')
        except Exception as e:
            raise e
        self.footprint_height = footprint_indices * z_size

    def export_to_excel(self, filepath):
        workbook = Workbook()

        worksheet = workbook.create_sheet(index_keyword, 0)
        workbook.remove(workbook['Sheet'])

        # Feed the indices
        m_blocks = self.structure.shape[0]
        n_blocks = self.structure.shape[1]
        for i in np.arange(m_blocks):
            for j in np.arange(n_blocks):
                cell = worksheet.cell(i + 1, j + 1)
                # TODO: Decoradores
                cell.value = self.footprint_indices[i, j]

        worksheet = workbook.create_sheet(height_keyword, 1)
        for i in np.arange(m_blocks):
            for j in np.arange(n_blocks):
                cell = worksheet.cell(i + 1, j + 1)
                # TODO: Decoradores
                cell.value = self.footprint_height[i, j]

        workbook.save(filepath)


