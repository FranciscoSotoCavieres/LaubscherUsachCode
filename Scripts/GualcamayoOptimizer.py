from itertools import takewhile
from Engine.CavingProductionPlanExtractionSpeedItem import CavingProductionPlanExtractionSpeedItem
from Engine.CavingProductionPlanTarget import CavingProductionPlanTarget
from Engine.CavingProductionPlanTargetItem import CavingProductionPlanTargetItem
from Engine.ExtractionPeriodBasicScheduleResult import ExtractionPeriodBasicScheduleResult
from Engine.ProductionPlanResult import ProductionPlanResult
import Models.Factory as ft
import Models.visualization as vis
import numpy as np
from Models.BlockModelDilution import BlockModelDilution
import os
import numpy as np
from Models.BlockModel import BlockModel
from Models.utils import accumulate_values, best_height_of_draw
from Models.Footprint import Footprint
import pandas as pd
import Models.utils as utils
import glob
from Scripts.script_utils import Column, Columns
import ezdxf
import ezdxf.entities
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from Engine.ProductionPlanEngine import ProductionPlanEngine
import pyvista as pv

au_name: str = 'AU_FINAL'
ore_name: str = 'ORE'
density_name: str = 'Density'
accumulate_value_name = 'accumulated_value'
value_name = 'value'


def load_data(block_model_csv_path: str, block_model_npy: str):
    block_model = ft.block_model_from_csv_file(
        block_model_csv_path, 'XC', 'YC', 'ZC', ',', [au_name, density_name, ore_name])

    density_data_set = block_model.get_data_set(density_name)
    density_data_set[density_data_set == 0] = 2.7
    block_model.update_dataset(density_name, density_data_set)
    block_model.save(block_model_npy)
    vis.draw_voxels(block_model=block_model,
                    data_set_name=au_name, min_value=2)


def dilute_block_models(block_model_npy_path: str, block_model_dilution_folder: str, from_z_index: int, to_z_index: int):

    # Insitu block model
    original_block_model = ft.block_model_from_npy_file(block_model_npy_path)

    # Crop the block model
    crop_block_model = ft.slice_block_model(
        original_block_model, to_z=to_z_index)

    # Dilute block model
    for i in range(from_z_index, to_z_index-10):

        file_name = os.path.join(
            block_model_dilution_folder, f'{i:03d} diluted.npy')
        level_block_model = ft.block_model_from_level(crop_block_model, i)
        block_model_dilution_engine = BlockModelDilution(
            50, level_block_model.structure)

        block_model_dilution_engine.compute_dilution_coefficients()

        diluted_block_model = block_model_dilution_engine.dilute_block_model(
            level_block_model, {au_name: 0.0,  density_name: 2.7, ore_name: 0})
        diluted_block_model.save(file_name)


def compute_best_height_level(folder):
    # Load the
    # columns = Columns.import_json(r"{folder}\all_columns")

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

    block_model_sample = ft.block_model_from_npy_file(
        fr'{folder}\000 diluted.npy')

    structure = block_model_sample.structure

    columns_df = pd.read_json(
        rf"{folder}\all_columns.json")

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
    df.to_csv(rf'{folder}\best_columns.csv')


def compute_tonnage_grade(folder: str):

    base_structure = ft.block_model_from_npy_file(
        os.path.join(folder, "000 diluted.npy")).structure

    x_size = base_structure.block_size[0]
    y_size = base_structure.block_size[1]

    df = pd.read_csv(rf'{folder}\best_columns.csv')
    x_centroids = np.array(df['x_centroids'])
    y_centroids = np.array(df['y_centroids'])
    column_values = np.array(df['column_values'])
    floors = np.array(df['floor'])
    heights = np.array(df['height'])
    slope_percentages = np.array(df['gradient_percentage'])

    tonnages: list[float] = []
    average_aus: list[float] = []

    for (x_centroid, y_centroid, floor, height) in zip(x_centroids, y_centroids, floors, heights):
        subscript = base_structure.get_subscripts(
            x_centroid, y_centroid, floor + base_structure.block_size[2]/2)
        i = subscript[0]
        j = subscript[1]
        k = subscript[2]

        block_height_index = int(height/base_structure.block_size[2])
        block_volume = base_structure.get_block_volume()

        current_block_model = ft.block_model_from_npy_file(
            os.path.join(folder, f'{k:03d} diluted.npy'))

        density = current_block_model.get_data_set(density_name)
        ore = current_block_model.get_data_set(ore_name)
        au = current_block_model.get_data_set(au_name)

        tonnage = utils.get_summation(0, block_height_index,
                                      density[i, j, :]) * block_volume
        average_au = utils.get_column_average(
            ore[i, j, :] * au[i, j, :], density[i, j, :], 0, block_height_index)

        tonnages.append(tonnage)
        average_aus.append(average_au)
        gold = tonnage * average_au

    df['tonnage'] = tonnages
    df['average_au'] = average_aus

    df.to_csv(os.path.join(folder, 'best_columns_tonnage.csv'))


def draw_grades(folder: str):

    base_structure = ft.block_model_from_npy_file(
        os.path.join(folder, "000 diluted.npy")).structure

    x_size = base_structure.block_size[0]
    y_size = base_structure.block_size[1]

    df = pd.read_csv(rf'{folder}\best_columns_tonnage.csv')
    x_centroids = np.array(df['x_centroids'])
    y_centroids = np.array(df['y_centroids'])
    column_values = np.array(df['column_values'])
    floors = np.array(df['floor'])
    heights = np.array(df['height'])
    slope_percentages = np.array(df['gradient_percentage'])
    tonnages = np.array(df['tonnage'])
    average_aus = np.array(df['average_au'])

    vis.export_footprint_mesh(x_centroids, y_centroids,
                              floors, heights, x_size, y_size, os.path.join(folder, "mesh.dxf"))

    plot = vis.draw_columns_from_arrays(np.array(x_centroids), np.array(y_centroids),
                                        np.array(floors), np.array(
                                            heights), x_size,
                                        y_size, np.array(column_values), np.array(slope_percentages), average_aus)


def compute_best_height(diluted_block_model_folder: str):
    overall_columns: Columns = Columns()
    for file_path in glob.glob(diluted_block_model_folder + r'\*.npy'):

        dilute_block_model = ft.block_model_from_npy_file(
            os.path.join(diluted_block_model_folder, file_path))
        diluted_structure = dilute_block_model.structure
        level = diluted_structure.get_left_down_near_corner()[2]

        valorization(dilute_block_model)

        footprint = best_height_of_draw(
            dilute_block_model, accumulate_value_name)
        accumulate_values = dilute_block_model.get_data_set(
            'accumulated_value')

        file_name = os.path.basename(file_path)
        footprint_path = rf'{diluted_block_model_folder}\{os.path.splitext(file_name)[0]}.xlsx'
        columns_path = rf'{diluted_block_model_folder}\{os.path.splitext(file_name)[0]}.json'
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

    overall_columns.export_json(
        rf'{diluted_block_model_folder}\all_columns.json')


def valorization(block_model: BlockModel):
    # Parameters
    value_set = value_name
    value_accumulated_set = accumulate_value_name

    au_key = au_name
    density_key = density_name
    ore_key = ore_name
    au_price = 1700 / 31.10345  # USD/g
    refining_cost = 7 / 31.10345  # USD/t
    mining_cost = 8  # USD/t
    processing_cost = 63  # USD/t
    development_cost = 3000  # USD/m²
    investment_cost = 2  # USD/t
    general_cost = 1  # USD/t
    recovery = .90  # fraction

    # Structure
    structure = block_model.structure
    block_volume = structure.get_block_volume()
    block_area = structure.get_block_area()
    shape = structure.shape

    # Get the datasets
    au = block_model.get_data_set(au_key)
    density = block_model.get_data_set(density_key)
    ore = block_model.get_data_set(ore_key)

    # Replace values
    au[au == -99] = 0
    density[density == -99] = 0
    density[density == 0] = 2.7

    # Compute tonnage
    tonnage = density * block_volume

    value_array = ((au_price - refining_cost) * au * ore * recovery -
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


def segment_squares(folder: str):
    document = ezdxf.readfile(os.path.join(folder, 'mesh_squares.dxf'))
    print(len(document.entities))

    base_structure = ft.block_model_from_npy_file(
        os.path.join(folder, "000 diluted.npy")).structure

    x_size = base_structure.block_size[0]
    y_size = base_structure.block_size[1]

    df = pd.read_csv(rf'{folder}\best_columns_tonnage.csv')
    x_centroids = np.array(df['x_centroids'])
    y_centroids = np.array(df['y_centroids'])
    column_values = np.array(df['column_values'])
    floors = np.array(df['floor'])
    heights = np.array(df['height'])
    average_aus = np.array(df['average_au'])
    tonnages = np.array(df['tonnage'])

    squares = list(filter(lambda entity: entity.DXFTYPE ==
                   'LWPOLYLINE', document.entities))

    current_index = 0
    for square in squares:

        # Start creating the first element
        points: list[Point] = []
        for point in square.get_points():
            points.append([point[0], point[1]])

        polygon = Polygon(points)

        current_x_centroids: list[float] = []
        current_y_centroids: list[float] = []
        current_column_values: list[float] = []
        current_floors: list[float] = []
        current_heights: list[float] = []
        current_average_aus: list[float] = []
        current_tonnages: list[float] = []

        for index in range(len(x_centroids)):
            x_centroid = x_centroids[index]
            y_centroid = y_centroids[index]
            column_value = column_values[index]
            floor = floors[index]
            height = heights[index]
            average_au = average_aus[index]
            tonnage = tonnages[index]

            test_point = Point(x_centroid, y_centroid)

            if (polygon.contains(test_point)):
                current_x_centroids.append(x_centroid)
                current_y_centroids.append(y_centroid)
                current_column_values.append(column_value)
                current_floors.append(floor)
                current_heights.append(height)
                current_average_aus.append(average_au)
                current_tonnages.append(tonnage)

        df = pd.DataFrame({
            'x_centroids': current_x_centroids,
            'y_centroids': current_y_centroids,
            'column_values': current_column_values,
            'floor': current_floors,
            'height': current_heights,
            'average_au': current_average_aus,
            'tonnage': current_tonnages
        })

        df.to_csv(os.path.join(
            folder, f'best column square {current_index}.csv'))
        current_index = current_index + 1


def stats_footprints(folder: str):
    average_aus = []
    tonnages = []
    areas = []

    for footprint_file in glob.glob(folder + "\Inclined Footprints\*.xlsx"):

        name = os.path.splitext(os.path.basename(footprint_file))[0]
        footprint_number = int(name.split(' ')[1])
        block_model_number = int(name.split(' ')[2])
        block_model_path = os.path.join(
            folder, f'{block_model_number:03d} diluted.npy')

        block_model = ft.block_model_from_npy_file(block_model_path)

        density_dataset = block_model.get_data_set(density_name)
        au_dataset = block_model.get_data_set(au_name)
        ore_dataset = block_model.get_data_set(ore_name)

        footprint = ft.footprint_from_excel(footprint_file, block_model)

        for i in np.arange(footprint.structure.shape[0]):
            for j in np.arange(footprint.structure.shape[1]):
                index = footprint.footprint_indices[i, j]
                if (index == 0):
                    continue

                average_au = utils.get_average(
                    0, index+1, au_dataset[i, j, :], density_dataset[i, j, :] * ore_dataset[i, j, :])

                tonnage = utils.get_summation(
                    0, index+1, density_dataset[i, j, :]) * 5*5*5

                average_aus.append(average_au)
                tonnages.append(tonnage)
                areas.append(1)

                print(f'{tonnage} {average_au}')

    average_au =np.sum(np.array(average_aus) * np.array(tonnages))/np.sum(np.array(tonnages))
    tonnage = np.sum(tonnages)
    area = 5*5 * np.sum(np.sum(areas))
    height = tonnage/ area / 2.7
    print('')



        


def build_inclined_footprints(folder: str):
    structure = ft.block_model_from_npy_file(
        os.path.join(folder, "000 diluted.npy")).structure

    x_centroids = []
    y_centroids = []
    floors = []
    heights = []
    benefits = []

    current_item = 1
    for columns_file in glob.glob(folder + "\*.csv"):
        if "best column square" not in columns_file:
            continue

        df = pd.read_csv(columns_file)
        if (len(df) == 0):
            continue
        mean_floor = (df['floor'] * df['height']).sum()/df['height'].sum()

        mean_floor = mean_floor - (mean_floor % 5)
        middle_block_height = structure.block_size[2] / 2
        level_subscript = structure.get_subscript(
            mean_floor + middle_block_height, 'k')

        x_centroids.extend(df['x_centroids'])
        y_centroids.extend(df['y_centroids'])
        floors.extend([mean_floor] * len(df['floor']))
        heights.extend([5] * len(df['floor']))
        benefits.extend([current_item] * len(df['floor']))
        current_item = current_item + 1

        # Prepare footprints

        # Gather the corresponding block model
        diluted_model = ft.block_model_from_npy_file(
            os.path.join(folder, f'{level_subscript:03d} diluted.npy'))

        valorization(diluted_model)

        accumulated_values = diluted_model.get_data_set(accumulate_value_name)
        new_accumulated_values = np.zeros(
            diluted_model.get_data_set(accumulate_value_name).shape)

        for x, y in zip(df['x_centroids'], df['y_centroids']):
            i = structure.get_subscript(x, 'i')
            j = structure.get_subscript(y, 'j')
            new_accumulated_values[i, j, :] = accumulated_values[i, j, :]

        diluted_model.update_dataset(
            accumulate_value_name, new_accumulated_values)

        footprint = best_height_of_draw(
            diluted_model, accumulate_value_name, 1)
        footprint.export_to_excel(os.path.join(
            folder, 'Inclined Footprints', f'footprint {current_item:03d} {level_subscript:03d}.xlsx'))

    # vis.draw_columns_from_arrays(
    #     x_centroids, y_centroids, floors, heights, 5, 5, benefits, benefits, benefits)
    vis.export_footprint_mesh(
        x_centroids, y_centroids, floors, heights, 5, 5, f'{folder}\mean_platforms.dxf')


def generate_plan(folder: str):

    footprint_folder = "Inclined Footprints"
    dump_folder = "dump"
    sequence_folder = "sequence"
    target_folder = "target"

    units: list[ExtractionPeriodBasicScheduleResult] = []

    plotter = pv.Plotter()
    current_item = 0
    number_of_periods = 50

    current_tonnage_target: dict[int, float] = {}

    for period in range(number_of_periods):
        current_tonnage_target[period] = 2 * 360 * 1000

    blocks_per_period = 10000/5/5

    files = list(glob.glob(f'{folder}\{footprint_folder}\*.xlsx'))

    # Central 2
    # Izquierda a derecha 13,14,15,16,17,4,5,6,7,8,9,
    # Derecha a izquierda 3, 10,11,12
    # sorted_footprints = [2,4,3,5,10,6,11,7,12,8,9,13,14,15,16,17]

    sorted_footprints = [2, 13, 3, 10, 14,
                         11, 15, 12, 16, 17, 4, 5, 6, 7, 8, 9]

    dxf_meshes: dict[int, ezdxf.entities.mesh.Mesh] = {}
    footprint_results: dict[int, ProductionPlanResult] = {}
    for footprint_number in sorted_footprints[:]:

        footprint_xlsx = list(filter(lambda file: int(
            os.path.basename(file).split(' ')[1]) == footprint_number, files))[0]

        name = os.path.splitext(os.path.basename(footprint_xlsx))[0]
        footprint_number = int(name.split(' ')[1])
        block_model_number = int(name.split(' ')[2])
        block_model_path = os.path.join(
            folder, f'{block_model_number:03d} diluted.npy')
        block_model = ft.block_model_from_npy_file(block_model_path)
        footprint = ft.footprint_from_excel(footprint_xlsx, block_model)

        target = ft.caving_configuration_from_excel(os.path.join
                                                    (folder, footprint_folder, target_folder, f'target {footprint_number:03d}.xlsx'))

        target.target_items.clear()

        for period in range(0, current_item):
            target.target_items.append(
                CavingProductionPlanTargetItem(period, current_tonnage_target[period], 0, 360))

        for period in range(current_item, number_of_periods):

            target.target_items.append(
                CavingProductionPlanTargetItem(period, current_tonnage_target[period], blocks_per_period, 360))

        # Start periods
        if (footprint_number == 2):
            target.target_items[0].incorporation_blocks = 8000/5/5
            target.target_items[0].target_tonnage = 0.5 * 1000 * 360
        if (footprint_number == 13):
            target.target_items[0].incorporation_blocks = 8000/5/5
            target.target_items[0].target_tonnage = 0.5 * 1000 * 360

        sequence = ft.sequence_from_excel(os.path.join
                                          (folder, footprint_folder, sequence_folder, f"sequence {footprint_number:03d}.xlsx"), block_model)

        production = ProductionPlanEngine(
            block_model, footprint, sequence, target)
        result = production.process()

        footprint_results[footprint_number] = result

        for period in range(number_of_periods):
            current_tonnage_target[period] = current_tonnage_target[period] - \
                result.period_results[period].tonnage
            if (np.abs(current_tonnage_target[period]) < 0.01):
                current_tonnage_target[period] = 100

        result.dump_units(os.path.join
                          (folder, footprint_folder, dump_folder, f"dump {footprint_number:03d}.csv"))
        result.export_excel(os.path.join
                            (folder, footprint_folder, dump_folder, f"dump {footprint_number:03d}.xlsx"))

        floor = block_model.structure.get_left_down_near_corner()[2]
        footprint_mesh = vis.get_mesh_from_schedule_units(
            result.units, block_model.structure, floor)

        dxf_meshes[footprint_number] = vis.get_dxf_mesh_from_mesh(
            footprint_mesh)

        plotter.add_mesh(footprint_mesh, smooth_shading=True,
                         split_sharp_edges=True, cmap='jet', clim=[0, 100])
        units.extend(result.units)
        current_item = current_item + 1

    df = pd.DataFrame({'period': [], 'tonnage': [], 'active_area': [
    ], 'incorporated_area': [], 'depleted_area': [], 'au': [], 'ore': []})

    for period in range(number_of_periods):
        period_tonnage_collection = []
        period_ore_collection = []
        period_au_collection = []
        period_active_area_collection = []
        period_depleted_area_collection = []
        period_incorporated_area_collection = []

        for footprint_result in footprint_results.values():

            period_active_area_collection.append(
                footprint_result.period_results[period].active_area_squared_meters)
            period_depleted_area_collection.append(
                footprint_result.period_results[period].depleted_area_squared_meters)
            period_incorporated_area_collection.append(
                footprint_result.period_results[period].incorporated_area_squared_meters)

            period_au_collection.append(
                footprint_result.period_results[period].average[au_name])
            period_ore_collection.append(
                footprint_result.period_results[period].average[ore_name])
            period_tonnage_collection.append(
                footprint_result.period_results[period].tonnage)

        tonnage = np.array(period_tonnage_collection).sum()
        active_area = np.array(period_active_area_collection).sum()
        depleted_area = np.array(period_depleted_area_collection).sum()
        incorporated_area = np.array(period_incorporated_area_collection).sum()

        average_au = (np.array(period_au_collection) *
                      np.array(period_tonnage_collection)).sum() / tonnage
        average_ore = (np.array(period_ore_collection) *
                       np.array(period_tonnage_collection)).sum() / tonnage

        df = df.append(pd.Series({'period': period, 'tonnage': tonnage, 'active_area': active_area, 'incorporated_area': incorporated_area,
                                  'depleted_area': depleted_area, 'au': average_au, 'ore': average_ore}), ignore_index=True)

    df.to_csv(os.path.join(folder, footprint_folder, dump_folder, 'pp.csv'))
    print(df)
    plotter.enable_parallel_projection()
    plotter.show()

    doc = ezdxf.new()
    modelspace = doc.modelspace()

    for period in dxf_meshes.keys():
        mesh = modelspace.add_mesh()

        with mesh.edit_data() as mesh_data:
            mesh_data.vertices = dxf_meshes[period].vertices
            mesh_data.faces = dxf_meshes[period].faces

    doc.saveas(os.path.join(folder, footprint_folder, dump_folder, 'pp.dxf'))
