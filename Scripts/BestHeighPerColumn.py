import Models.Factory as ft
import Models.visualization as vis
import numpy as np
from Models.BlockModelDilution import BlockModelDilution
import os
import numpy as np
from Models.BlockModel import BlockModel
from Models.utils import accumulate_values, best_height_of_draw
from Models.Footprint import Footprint
import json
import pandas as pd
import vedo
import Models.utils as utils


class Column:
    value: float
    level_meters: float
    height_meters: float
    subscript_i: int
    subscript_j: int
    x: float
    y: float

    def __init__(self, value: float, level_meters: float, height_meters: float,
                 subscript_i: int, subscript_j: int, x: float, y: float) -> None:
        self.value = value
        self.level_meters = level_meters
        self.height_meters = height_meters
        self.subscript_i = subscript_i
        self.subscript_j = subscript_j
        self.x = x
        self.y = y


class Columns:
    storage: list[Column]

    def __init__(self) -> None:
        self.storage = []

    def add_column(self, column: Column):
        self.storage.append(column)

    def export_json(self, file_path: str):
        items: list[dict[str, float]] = []

        for column in self.storage:
            items.append({
                "value": column.value,
                "level_meters": column.level_meters,
                "height_meters": column.height_meters,
                "subscript_i": column.subscript_i,
                "subscript_j": column.subscript_j,
                "x": column.x,
                "y": column.y})

        with open(file_path, "w") as write:
            json.dump(items, write)

    def import_json(file_path: str):
        columns = Columns()
        with open(file_path, 'r') as read:
            items: list[dict[str, float]] = json.loads(read.readline())

        for item in items:
            value = item['value']
            level_meters = item['level_meters']
            height_meters = item['height_meters']
            subscript_i = item['subscript_i']
            subscript_j = item['subscript_j']
            x = item['x']
            y = item['y']
            columns.add_column(Column(value, level_meters,
                               height_meters, subscript_i, subscript_j, x, y))
        return columns


def load_data(block_model_csv_path: str, block_model_npy: str):

    block_model = ft.block_model_from_csv_file(
        block_model_csv_path, 'centroid_x', 'centroid_y', 'centroid_z', ',', ['cut', 'ag', 'dens',])
    block_model.save(block_model_npy)


def init_study(block_model_npy_path: str, block_model_dilution_folder: str, sector: str, to_z: int):

    original_block_model = ft.block_model_from_npy_file(
        block_model_npy_path)

    # vis.draw_voxels(original_block_model,'cut')

    # return

    # Crop the block model
    crop_block_model = ft.slice_block_model(original_block_model, to_z=to_z)

    # Dilute block model
    for i in range(to_z-10):

        file_name = block_model_dilution_folder + \
            f"\{sector} {i:000} diluted.npy"
        level_block_model = ft.block_model_from_level(crop_block_model, i)
        block_model_dilution_engine = BlockModelDilution(
            50, level_block_model.structure)

        block_model_dilution_engine.compute_dilution_coefficients()

        diluted_block_model = block_model_dilution_engine.dilute_block_model(
            level_block_model, {"cut": 0.0, "ag": 0.0, "dens": 2.7})
        diluted_block_model.exists_data_set('cut')
        diluted_block_model.save(file_name)


def draw_grades(folder: str):
    # Load the
    # columns = Columns.import_json(r"{folder}\all_

    sectors = [r'C', r'SO', r'NO', r'E']
    block_model_folder_names = [r'\Diluted Block Model C', r'\Diluted Block Model SO',
                                r'\Diluted Block Model NO', r'\Diluted Block Model E']

    x_size: float = 5
    y_size: float = 5

    x_centroids = []
    y_centroids = []
    subscript_i_collection = []
    subscript_j_collection = []
    y_centroids = []
    floors = []
    heights = []
    column_values = []
    slope_percentages = []

    for (sector, block_model_folder) in zip(block_model_folder_names, sectors):

        block_model_sample = ft.block_model_from_npy_file(
            fr'{folder}\{block_model_folder}\base_diluted.npy')

        structure = block_model_sample.structure

        heights_df = pd.read_json(
            rf"{folder}{block_model_folder}\best_columns.csv")

        grades_df = pd.read_json(
            rf"{folder}{block_model_folder}\{sector} cut and tonnage.csv")

        group_by_column = heights_df.groupby(["subscript_i", "subscript_j"])

        indices = np.zeros(structure.shape[:2], dtype=np.int32)

        current_subscripts_i = []
        current_subscripts_j = []

        for (key, values) in group_by_column:
            index = values.idxmax()['value']
            column = heights_df.iloc[[index]]

            value = column['value'][index]
            x = column['x'][index]
            y = column['y'][index]
            subscript_i = column['subscript_i'][index]
            subscript_j = column['subscript_j'][index]

            current_subscripts_i.append(subscript_i)
            current_subscripts_j.append(subscript_j)

            subscript_i_collection.append(subscript_i)
            subscript_j_collection.append(subscript_j)

            level_in_meters = column['level_meters'][index]
            height_in_meters = column['height_meters'][index]

            indices[subscript_i, subscript_j] = level_in_meters / \
                structure.block_size[2]

            x_centroids.append(x)
            y_centroids.append(y)
            column_values.append(value)
            floors.append(level_in_meters)
            heights.append(height_in_meters)

        footprint = Footprint(indices, structure)
        slope = footprint.compute_slope_percentage()
        for (subscript_i, subscript_j) in zip(current_subscripts_i, current_subscripts_j):
            slope_percentages.append(slope[subscript_i, subscript_j])

    data = {'x_centroids': x_centroids, 'y_centroids': y_centroids,
            'column_values': column_values, 'floor': floors, 'subscript_i': subscript_i_collection,
            'subscript_j': subscript_j_collection, 'height': heights, 'gradient_percentage': slope_percentages}

    df = pd.DataFrame(data)
    # df.to_csv(rf'{folder}\best_columns_E.csv')

    plot = vis.draw_columns_from_arrays(np.array(x_centroids), np.array(y_centroids),
                                        np.array(floors), np.array(
                                            heights), x_size,
                                        y_size, np.array(column_values), np.array(slope_percentages))


def compute_tonnage_and_grade(folder):

    sectors = [r'C', r'SO', r'NO', r'E']
    block_model_folder_names = [r'Diluted Block Model C', r'Diluted Block Model SO',
                                r'Diluted Block Model NO', r'Diluted Block Model E']

    all_total_tonnage = 0
    all_total_tonnage_per_cut = 0
    for (sector, block_model_folder) in zip(sectors, block_model_folder_names):

        df = pd.read_csv(
            rf"{folder}\{block_model_folder}\{sector} cut and tonnage.csv")

        current_tonnage = df['tonnage'].sum()
        current_tonnage_per_cut = (df['tonnage'] * df['cut']).sum()

        all_total_tonnage = all_total_tonnage + current_tonnage
        all_total_tonnage_per_cut = all_total_tonnage_per_cut + current_tonnage_per_cut
        cut = current_tonnage_per_cut/current_tonnage

        print(f'{sector},{cut},{current_tonnage/1e6}')

        continue
        data_frame_path = rf"{folder}\{block_model_folder}\best_columns.csv"
        df = pd.read_csv(data_frame_path)

        structure = ft.block_model_from_npy_file(
            rf"{folder}\{block_model_folder}\base_diluted.npy").structure

        overall_floor = structure.get_left_down_near_corner()[2]
        block_height = structure.block_size[2]
        x_size = structure.block_size[0]
        y_size = structure.block_size[1]

        area = x_size * y_size

        tonnage_collection = []
        cut_collection = []
        subscript_i_collecion = []
        subscript_j_collecion = []

        for (index, item) in df.iterrows():

            height = item['height']
            floor = item['floor']
            subscript_i = int(item['subscript_i'])
            subscript_j = int(item['subscript_j'])

            level = int((floor - overall_floor) / block_height)
            index = int(height / block_height)

            current_block_model_path = fr"{folder}\{block_model_folder}\{sector} {level} diluted.npy"
            block_model_diluted = ft.block_model_from_npy_file(
                current_block_model_path)
            cut = block_model_diluted.get_data_set('cut')
            dens = block_model_diluted.get_data_set('dens')

            average_cut = utils.get_column_average(cut[subscript_i, subscript_j, :],
                                                   dens[subscript_i, subscript_j, :], 0, index+1)

            average_dens = utils.get_column_average(dens[subscript_i, subscript_j, :],
                                                    dens[subscript_i, subscript_j, :], 0, index+1)

            tonnage = height * area * average_dens

            print(f"{average_cut},{tonnage}")

            tonnage_collection.append(tonnage)
            cut_collection.append(average_cut)
            subscript_i_collecion.append(subscript_i)
            subscript_j_collecion.append(subscript_j)

        new_data = {"cut": cut_collection, "tonnage": tonnage_collection,
                    "subscript_i": subscript_i_collecion,
                    "subscript_j": subscript_j_collecion}
        new_df = pd.DataFrame(new_data)
        new_df.to_csv(
            rf"{folder}\{block_model_folder}\{sector} cut and tonnage.csv")
    average_cut = all_total_tonnage_per_cut/all_total_tonnage
    all_tonnage = all_total_tonnage


def compute_best_height_level(folder):
    # Load the
    # columns = Columns.import_json(r"{folder}\all_columns")

    block_model_folder_names = [r'\Diluted Block Model C', r'\Diluted Block Model SO',
                                r'\Diluted Block Model NO', r'\Diluted Block Model E']

    x_size: float = 5
    y_size: float = 5

    x_centroids = []
    y_centroids = []
    subscript_i_collection = []
    subscript_j_collection = []
    y_centroids = []
    floors = []
    heights = []
    column_values = []
    slope_percentages = []

    for block_model_folder in block_model_folder_names[:]:

        block_model_sample = ft.block_model_from_npy_file(
            fr'{folder}\{block_model_folder}\base_diluted.npy')

        structure = block_model_sample.structure

        columns_df = pd.read_json(
            rf"{folder}{block_model_folder}\all_columns.json")

        group_by_column = columns_df.groupby(["subscript_i", "subscript_j"])

        indices = np.zeros(structure.shape[:2], dtype=np.int32)

        current_subscripts_i = []
        current_subscripts_j = []

        for (key, values) in group_by_column:
            index = values.idxmax()['value']
            column = columns_df.iloc[[index]]

            value = column['value'][index]
            x = column['x'][index]
            y = column['y'][index]
            subscript_i = column['subscript_i'][index]
            subscript_j = column['subscript_j'][index]

            current_subscripts_i.append(subscript_i)
            current_subscripts_j.append(subscript_j)

            subscript_i_collection.append(subscript_i)
            subscript_j_collection.append(subscript_j)

            level_in_meters = column['level_meters'][index]
            height_in_meters = column['height_meters'][index]

            indices[subscript_i, subscript_j] = level_in_meters / \
                structure.block_size[2]

            x_centroids.append(x)
            y_centroids.append(y)
            column_values.append(value)
            floors.append(level_in_meters)
            heights.append(height_in_meters)

        footprint = Footprint(indices, structure)
        slope = footprint.compute_slope_percentage()
        for (subscript_i, subscript_j) in zip(current_subscripts_i, current_subscripts_j):
            slope_percentages.append(slope[subscript_i, subscript_j])

    data = {'x_centroids': x_centroids, 'y_centroids': y_centroids,
            'column_values': column_values, 'floor': floors, 'subscript_i': subscript_i_collection,
            'subscript_j': subscript_j_collection, 'height': heights, 'gradient_percentage': slope_percentages}

    df = pd.DataFrame(data)
    # df.to_csv(rf'{folder}\best_columns_E.csv')

    plot = vis.draw_columns_from_arrays(np.array(x_centroids), np.array(y_centroids),
                                        np.array(floors), np.array(
                                            heights), x_size,
                                        y_size, np.array(column_values), np.array(slope_percentages))


def compute_best_height(block_model_folder: str):
    overall_columns: Columns = Columns()
    for file_name in os.listdir(block_model_folder):
        if (os.path.splitext(file_name)[-1] != '.npy'):
            continue

        file_path = rf'{block_model_folder}\{file_name}'
        dilute_block_model = ft.block_model_from_npy_file(file_path)
        diluted_structure = dilute_block_model.structure
        level = diluted_structure.get_left_down_near_corner()[2]

        valorization(dilute_block_model)

        footprint = best_height_of_draw(
            dilute_block_model, 'accumulated_value')
        accumulate_values = dilute_block_model.get_data_set(
            'accumulated_value')

        footprint_path = rf'{block_model_folder}\{os.path.splitext(file_name)[0]}.xlsx'
        columns_path = rf'{block_model_folder}\{os.path.splitext(file_name)[0]}.json'
        footprint.export_to_excel(footprint_path)

        columns = Columns()
        for i in range(footprint.footprint_height.shape[0]):
            for j in range(footprint.footprint_height.shape[1]):
                height = footprint.footprint_height[i, j]
                index = footprint.footprint_indices[i, j]
                if (index == 0):
                    continue
                value = accumulate_values[i, j, index-1]
                centroid = diluted_structure.get_centroid(i, j, 0)
                columns.add_column(
                    Column(value, level, height, i, j, centroid[0], centroid[1]))
                overall_columns.add_column(
                    Column(value, level, height, i, j, centroid[0], centroid[1]))
        columns.export_json(columns_path)

    overall_columns.export_json(rf'{block_model_folder}\all_columns.json')


def valorization(block_model: BlockModel):
    # Parameters
    value_set = 'value'
    value_accumulated_set = 'accumulated_value'

    cut_key = "cut"
    density_key = "dens"
    copper_price = 3.3 * 2204.5  # USD/t
    refining_cost = .39 * 2204.5  # USD/t
    mining_cost = 8  # USD/t
    processing_cost = 11.52  # USD/t
    development_cost = 3000  # USD/m²
    investment_cost = 2  # USD/t
    general_cost = 1  # USD/t

    # Structure
    structure = block_model.structure
    block_volume = structure.get_block_volume()
    block_area = structure.get_block_area()
    shape = structure.shape

    # Get the datasets
    cut = block_model.get_data_set(cut_key)

    density = block_model.get_data_set(density_key)

    # Replace values
    cut[cut == -99] = 0
    density[density == -99] = 0
    density[density == 0] = 2.7

    # Compute tonnage
    tonnage = density * block_volume

    value_array = ((copper_price - refining_cost) * cut / 100 -
                   (mining_cost + processing_cost+investment_cost+general_cost)) * tonnage

    for i in np.arange(shape[0]):
        for j in np.arange(shape[1]):
            value_array[i, j, 0] -= development_cost * block_area

    # Check if exists the data set
    if block_model.exists_data_set(value_set):
        block_model.update_dataset(value_set, value_array)
    else:
        block_model.add_dataset(value_set, value_array)

    accumulated_value_array = accumulate_values(value_array)

    if block_model.exists_data_set(value_accumulated_set):
        block_model.update_dataset(
            value_accumulated_set, accumulated_value_array)
    else:
        block_model.add_dataset(value_accumulated_set, accumulated_value_array)
