from io import StringIO
import numpy as np
from math import cos, sin, radians
from Models.BlockModel import BlockModel
from Models.BlockModelStructure import BlockModelStructure
from Models.Footprint import Footprint
from Models.Sequence import Sequence
from typing import Tuple


def polygon_area(x: np.ndarray, y: np.ndarray):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def normalize_topography(absolute_topography: np.ndarray, structure: BlockModelStructure) -> np.ndarray:
    return (absolute_topography - structure.offset[2] + structure.block_size[2]) / structure.block_size[2]


def best_height_of_draw(block_model: BlockModel, data_set_name: str, min_blocks: int = None, max_blocks: int = None,
                        min_value: float = 0):
    values = block_model.get_data_set(data_set_name)
    structure = block_model.structure
    x_blocks = structure.shape[0]
    y_blocks = structure.shape[1]

    footprint_indices = np.zeros([x_blocks, y_blocks]).astype('int')

    for i in np.arange(x_blocks):
        for j in np.arange(y_blocks):
            index = best_height_of_draw_column(
                values[i, j, :], min_blocks, max_blocks, min_value) + 1
            footprint_indices[i, j] = index

    footprint = Footprint(footprint_indices, structure)
    return footprint


def get_column_average(column_values: np.ndarray, columns_weights: np.ndarray = None, min_block_index=None,
                       max_block_index=None):
    min_index = 0
    max_index = column_values.shape[2]

    if min_index is not None:
        min_index = min_block_index

    if max_index is not None:
        max_index = max_block_index

    new_column_values: np.ndarray = column_values[min_index:max_index]

    if column_values is None:
        average = np.average(new_column_values)
    else:
        new_column_weights = columns_weights[min_index:max_index]
        average = np.average(new_column_values, new_column_weights)

    return average


def best_height_of_draw_column(column: np.ndarray, min_blocks: int = None, max_blocks: int = None,
                               min_value: float = 0) -> int:
    # Analyze the location
    min_index = 0
    max_index = column.shape[0]

    if min_blocks is not None:
        min_index = min_blocks
    if max_blocks is not None:
        max_index = max_blocks

    max_value = np.max(column[min_index:max_index])
    if max_value <= min_value:
        return -1

    index = -1
    for i in np.arange(min_index, max_index):
        if column[i] == max_value:
            index = i
            break
    return index


def accumulate_values(array: np.ndarray) -> np.ndarray:
    # Accumulate values
    accumulate_array = np.zeros(array.shape)
    accumulate_array[:, :, 0] = array[:, :, 0]
    for i in np.arange(1, array.shape[2]):
        accumulate_array[:, :, i] = accumulate_array[:,
                                                     :, i - 1] + array[:, :, i]
    return accumulate_array


def reduce_csv_file_variables(filepath: str, out_filepath: str, variables: list[str], separator=','):
    # Get the header
    with open(filepath, 'r') as read_file:
        header = read_file.readline().strip()

    # Search for the indices
    header_splits = header.split(separator)
    index_header: dict[str, int] = dict([])
    indices: list[int] = []
    for variable in variables:
        index = header_splits.index(variable)
        index_header[variable] = index
        indices.append(index)

    def build_elements(items: list[str]) -> str:
        builder = StringIO()
        for item in items:
            builder.write(item)
            builder.write(',')
        output = builder.getvalue()[:-1]
        output = output + '\n'
        return output

    # Add items
    with open(filepath, 'r') as read_file:
        with open(out_filepath, 'w+') as write_file:
            while True:
                line = read_file.readline()
                if not line:
                    break
                line_splits = line.strip().split(separator)
                values: list[str] = []
                for index in indices:
                    values.append(line_splits[index])
                write_file.writelines(build_elements(values))


def rotate_point2d(point: np.ndarray, angle_degrees: float, origin: np.ndarray = np.array([0, 0])):
    x = point[0]
    y = point[1]

    ox = origin[0]
    oy = origin[1]

    angle_radians = radians(angle_degrees)

    qx = ox + cos(angle_radians) * (x - ox) + sin(angle_radians) * (y - oy)
    qy = oy + -sin(angle_radians) * (x - ox) + cos(angle_radians) * (y - oy)

    return np.array([qx, qy])


def sequence_footprint(footprint: Footprint, azimuth_degrees: float) -> Sequence:
    structure = footprint.structure
    footprint_indices = footprint.footprint_indices
    corner = structure.get_left_down_near_corner()
    distance_subscripts: dict[Tuple[int, int], float] = dict()
    for i in np.arange(structure.shape[0]):
        for j in np.arange(structure.shape[1]):
            if footprint_indices[i,j] <= 0:
                continue
            centroid = structure.get_centroid(i, j, 0)
            centroid = rotate_point2d(centroid, -azimuth_degrees+90, corner)
            distance_subscripts[(i, j)] = centroid[0]
    distance_subscripts = {k: v for k, v in sorted(
        distance_subscripts.items(), key=lambda item: item[1])}

    # Excel File
    sequence_indices = np.ones([structure.shape[0], structure.shape[1]]) * -1
    current_index = 1
    for key in distance_subscripts.keys():
        i, j = key
        sequence_indices[i, j] = current_index
        current_index = current_index + 1

    sequence = Sequence(sequence_indices, structure)
    return sequence
