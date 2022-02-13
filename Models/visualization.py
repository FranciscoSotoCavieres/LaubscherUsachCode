from numpy import mgrid
import numpy as np
import pyvista as pv
from Models.BlockModel import BlockModel


def grid(block_model: BlockModel, data_set: str):
    values = block_model.get_data_set(data_set)
    structure = block_model.structure
    grid = pv.UniformGrid()

    grid.dimensions = np.array(values.shape) + 1

    # References
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (structure.block_size[0], structure.block_size[1], structure.block_size[2])

    # Add the data values to the cell data
    grid.cell_data["values"] = values.flatten(order="F")

    grid.contour().plot(show_bounds=True)

    # Now plot the grid!
    # grid.plot(show_edges=True)


def contours(block_model: BlockModel, data_set: str):
    values = block_model.get_data_set(data_set)
    structure = block_model.structure

    start_x = structure.offset[0] - structure.block_size[0]
    end_x = start_x + (structure.shape[0]) * structure.block_size[0]
    delta_x = structure.block_size[0]

    start_y = structure.offset[1] - structure.block_size[1]
    end_y = start_y + (structure.shape[1]) * structure.block_size[1]
    delta_y = structure.block_size[1]

    start_z = structure.offset[2] - structure.block_size[2]
    end_z = start_z + (structure.shape[2]) * structure.block_size[2]
    delta_z = structure.block_size[2]

    (x, y, z) = mgrid[start_x:end_x:delta_x, start_y:end_y:delta_y, start_z:end_z:delta_z]
    grid = pv.StructuredGrid(x, y, z)

    grid['vol'] = values.flatten(order='F')
    grid.contour([0.5, 0.7, 1.2]).plot()

def draw_footprint():
    pass
