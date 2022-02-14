import unittest
import numpy as np
import pandas as pd
import pyvista as pv
import os
from Models.BlockModel import BlockModel
from Models.BlockModelDilution import BlockModelDilution
from Models.BlockModelStructure import BlockModelStructure
from Models.visualization import contours, draw_footprint
from Models.utils import accumulate_values, best_height_of_draw, sequence
import Models.Factory as ft


class BlockModelShould(unittest.TestCase):
    def test_block_model_load(self):
        # load data
        file_path = f'{os.getcwd()}/test_data/G8.csv'
        data = pd.read_csv(file_path, ',')
        print(data.keys())

        x = data['X']
        y = data['Y']
        z = data['Z']

        blockModel = ft.block_model_from_csv_file(file_path, 'X', 'Y', 'Z')

        values = blockModel.get_data_set('Cut')

        # Create the spatial reference
        grid = pv.UniformGrid()

        # Set the grid dimensions: shape + 1 because we want to inject our values on
        #   the CELL data
        grid.dimensions = np.array(values.shape) + 1

        # Edit the spatial reference
        grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
        grid.spacing = (20, 20, 18)  # These are the cell sizes along each axis

        # Add the data values to the cell data
        grid.cell_data["values"] = values.flatten(
            order="F")  # Flatten the array!

        # Now plot the grid!
        grid.plot(show_edges=True)

        self.assertEqual(True, True)  # add assertion here

    def test_block_model_dilution(self):
        # load data
        file_path = f'{os.getcwd()}/test_data/G8.csv'
        data = pd.read_csv(file_path, ',')
        print(data.keys())

        x = data['X']
        y = data['Y']
        z = data['Z']
        structure = BlockModelStructure.from_xyz(
            np.array(x), np.array(y), np.array(z))

        block_model = ft.block_model_from_csv_file(file_path, 'X', 'Y', 'Z')

        dilution = BlockModelDilution(10, structure)
        dilution.compute_dilution_coefficients()

        values = dilution.dilute_dataset(block_model.get_data_set('Cut'))

        new_block_model = ft.block_model_from_level(block_model, 4)
        filepath = r'C:\Users\franc\Desktop\myFile.npy'
        new_block_model.save(filepath)

        block_model = ft.block_model_from_file(filepath)
        contours(block_model, 'Cut')

    def test_accumulate_values(self):
        block_model = self.get_block_model_from_py_file()
        array = np.ones(block_model.structure.shape)
        accumulated = accumulate_values(array)

        assert accumulated[0, 0, 0] == 1, "Should be 1"
        assert accumulated[0, 0, 1] == 2, "Should be 2"

    def test_footprint_construction(self):
        block_model = self.get_block_model_from_py_file()
        value_set = 'value'
        accumulated_value_set = 'accumulated_value'
        self.valorization(block_model, value_set, accumulated_value_set)

        footprint = best_height_of_draw(block_model, accumulated_value_set)
        footprint.export_to_excel(f'{os.getcwd()}/test_data/EXCEL.xlsx')

        read_footprint = ft.footprint_from_excel(
            f'{os.getcwd()}/test_data/EXCEL.xlsx', block_model)
        current_sequence = sequence(read_footprint, 180)

        current_sequence.export_to_excel(
            f'{os.getcwd()}/test_data/Sequence.xlsx')

        draw_footprint(read_footprint, block_model,
                       current_sequence.sequence_indices)

    def get_block_model_from_csv(self):
        file_path = f'{os.getcwd()}/test_data/G8.csv'
        data = pd.read_csv(file_path, ',')
        block_model = ft.block_model_from_csv_file(file_path, 'X', 'Y', 'Z')
        return block_model

    def get_block_model_from_py_file(self):
        block_model = ft.block_model_from_file(
            f'{os.getcwd()}/test_data/G8.npy')
        return block_model

    # noinspection DuplicatedCode
    def valorization(self, block_model: BlockModel, value_set: str, value_accumulated_set: str):
        # Parameters
        copper_price = 3.2 * 2204.5  # USD/t
        refining_cost = .15 * 2204.5  # USD/t
        mining_cost = 8  # USD/t
        processing_cost = 8  # USD/t
        development_cost = 3000  # USD/m²

        # Structure
        structure = block_model.structure
        block_volume = structure.get_block_volume()
        block_area = structure.get_block_area()
        shape = structure.shape

        # Get the datasets
        cut = block_model.get_data_set('Cut')
        density = block_model.get_data_set('Density')

        # Compute tonnage
        tonnage = density * block_volume

        value_array = ((copper_price - refining_cost) * cut /
                       100 - (mining_cost + processing_cost)) * tonnage

        for i in np.arange(shape[0]):
            for j in np.arange(shape[1]):
                value_array[i, j, 0] -= development_cost * block_area

        # Check if exists the data set
        if block_model.exits_data_set(value_set):
            block_model.update_dataset(value_set, value_array)
        else:
            block_model.add_dataset(value_set, value_array)

        accumulated_value_array = accumulate_values(value_array)

        if block_model.exits_data_set(value_accumulated_set):
            block_model.update_dataset(
                value_accumulated_set, accumulated_value_array)
        else:
            block_model.add_dataset(
                value_accumulated_set, accumulated_value_array)


if __name__ == '__main__':
    unittest.main()
