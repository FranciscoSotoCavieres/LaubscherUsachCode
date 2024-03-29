import numpy as np
from Models.BlockModel import BlockModel
from Models.utils import accumulate_values


def valorization(block_model: BlockModel):
    # Parameters
    value_set = 'value'
    value_accumulated_set = 'accumulated_value'

    cut_key = "Cut"
    density_key = "Density"
    recovery_key = "Recu"
    copper_price = 3.2 * 2204.5  # USD/t
    refining_cost = .15 * 2204.5  # USD/t
    mining_cost = 8  # USD/t
    processing_cost = 8  # USD/t
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
    recovery = block_model.get_data_set(recovery_key)

    # Compute tonnage
    tonnage = density * block_volume

    value_array = ((copper_price - refining_cost) * cut / 100 * recovery / 100 -
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
