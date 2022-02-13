import numpy as np
from openpyxl import Workbook
from openpyxl.styles.borders import Border, Side

from Constants.UsachColors import color3


def get_border(border_type: str = 'thin', color: str = color3):
    side = Side(border_type=border_type, color=color)
    border = Border(left=side,right=side,bottom=side,top=side)
    return border


def export_matrix(array2d: np.ndarray, workbook: Workbook, sheet_name: str):
    worksheet = workbook.create_sheet(sheet_name, 0)
    # Feed the indices
    m_blocks = array2d.shape[0]
    n_blocks = array2d.shape[1]
    for i in np.arange(m_blocks):
        for j in np.arange(n_blocks):
            cell = worksheet.cell(i + 1, j + 1)
            cell.border =
            cell.style.
            # TODO: Decoradores
            if self.sequence_indices[i, j] == -1:
                cell.value = None
            else:
                cell.value = self.sequence_indices[i, j]
    workbook.save(filepath)


def remove_default_worksheet(workbook: Workbook):
    """
    It has to have at least 1 other worksheet
    :param workbook: Workbook
    """
    workbook.remove(workbook['Sheet'])
