import numpy as np
from openpyxl import Workbook
from Models.BlockModelStructure import BlockModelStructure

sequence_keyword = 'Sequence'

class Sequence():
    structure: BlockModelStructure
    sequence_indices: np.ndarray

    def __init__(self, sequence_indices: np.ndarray, structure: BlockModelStructure):
        self.sequence_indices = sequence_indices.astype('int')
        self.structure = structure

        x_blocks = self.structure.shape[0]
        y_blocks = self.structure.shape[1]

        # Check dimensions
        try:
            # Check dimensions
            if self.sequence_indices.shape[0] != x_blocks or self.sequence_indices.shape[1] != y_blocks:
                raise Exception('Block model and sequence dimensions doesn''t match')
        except Exception as e:
            raise e

        # Put in -1 the non sequenced
        self.sequence_indices[self.sequence_indices < 0]  = -1

    def export_to_excel(self, filepath):
        workbook = Workbook()

        worksheet = workbook.create_sheet(sequence_keyword, 0)
        workbook.remove(workbook['Sheet'])

        # Feed the indices
        m_blocks = self.structure.shape[0]
        n_blocks = self.structure.shape[1]
        for i in np.arange(m_blocks):
            for j in np.arange(n_blocks):
                cell = worksheet.cell(i + 1, j + 1)
                # TODO: Decoradores
                if self.sequence_indices[i, j] == -1:
                    cell.value = None
                else:
                    cell.value = self.sequence_indices[i, j]
        workbook.save(filepath)
