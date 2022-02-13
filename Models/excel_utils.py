import numpy as np
from openpyxl import Workbook, worksheet
from openpyxl.utils import get_column_letter
from openpyxl.styles.borders import Border, Side, BORDER_THICK

from Constants.UsachColors import COLOR_2


def get_border(border_type=BORDER_THICK, color: str = COLOR_2):
    side = Side(border_style=border_type, color=color)
    border = Border(left=side, right=side, bottom=side, top=side)
    return border


def export_matrix(array2d: np.ndarray, workbook: Workbook, sheet_name: str, filepath: str, none_value: float = None):
    worksheet = workbook.create_sheet(sheet_name, 0)
    # Feed the indices
    m_blocks = array2d.shape[0]
    n_blocks = array2d.shape[1]
    for i in np.arange(m_blocks):
        for j in np.arange(n_blocks):
            cell = worksheet.cell(m_blocks - i, j + 1)
            cell.border = get_border()
            value = array2d[i, j]
            if (none_value == None):
                cell.value = value
            else:
                if (value == none_value):
                    cell.value = None
                else:
                    cell.value = value

    for i in np.arange(m_blocks):
        worksheet.row_dimensions[i].height = 30
    for j in np.arange(n_blocks):
        worksheet.column_dimensions[i].width = get_column_letter(j)
    workbook.save(filepath)


def load_matrix(workbook: Workbook, sheet_name: str, rows: int, columns=int):
    sheet: worksheet = workbook[sheet_name]
    array = np.zeros([rows, columns])
    for i in np.arange(rows):
        for j in np.arange(columns):
            value = int(sheet.cell(rows - i, j + 1).value)
            array[i, j] = value
    return array


def remove_default_worksheet(workbook: Workbook):
    """
    It has to have at least 1 other worksheet
    :param workbook: Workbook
    """
    workbook.remove(workbook['Sheet'])
