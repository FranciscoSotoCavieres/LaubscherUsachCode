import numpy as np
from openpyxl import Workbook
from Models.BlockModelStructure import BlockModelStructure
from Models.excel_utils import export_matrix, remove_default_worksheet

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
                raise Exception(
                    'Block model and sequence dimensions doesn''t match')
        except Exception as e:
            raise e

        # Put in -1 the non sequenced
        self.sequence_indices[self.sequence_indices < 0] = -1

    def export_to_excel(self, filepath):
        workbook = Workbook()
        export_matrix(self.sequence_indices, workbook,
                      sequence_keyword,  -1)
        remove_default_worksheet(workbook)
        workbook.save(filepath)
