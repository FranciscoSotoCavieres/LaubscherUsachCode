from Models.BlockModelStructure import BlockModelStructure
from Models.Footprint import Footprint
from Models.BlockModel import BlockModel
from numpy import mgrid
import numpy as np
import pyvista as pv
from pyvista import PolyData, plot, themes
from typing import Tuple
import matplotlib as plt
import vedo

__faces_per_cube = 12
__vertices_per_cube = 8
__scalars_per_cube = 12
__scale_factor = .8


def blockmodel_view(block_model: BlockModel, data_set: str, Discrete: bool = None):
    '''
    shows the block model in some particular dataset

            Parameters:
                block_model (BlockModel): object with block model information
                data_set (str): Database parameter to display. Example: 'Cut'
                Discrete (bool, optional):  Default is False. If data_set is discrete variable put True
    '''

    Data = block_model.dataset[data_set]
    Origin = block_model.structure.offset
    blockSize = block_model.structure.block_size
    Dimension = block_model.structure.shape + 1

    p = pv.Plotter()
    grid = pv.UniformGrid()

    grid.dimensions = (Data.shape[0]+1, Data.shape[1]+1, Data.shape[2]+1)
    grid.spacing = block_model.structure.block_size
    grid.origin = block_model.structure.offset
    grid.cell_arrays[data_set] = Data.flatten(order='F')

    threshed = grid.threshold([-999999, 99999])

    if (Discrete == None) or (Discrete == False):
        cmap = plt.cm.get_cmap("jet")
        p.add_mesh(threshed, cmap=cmap, show_edges=True)  # ,rng=[0,1])

    else:
        colors = ['blue', 'orange', 'green', 'red', 'purple',
                  'brown', 'pink', 'gray', 'olive', 'cyan', 'yellow']
        i = 0
        for valor in np.unique(Data):
            threshed = grid.threshold([valor, valor])
            p.add_mesh(threshed, color=colors[i],
                       show_edges=True, label=str(valor))
            i += 1
        p.add_legend(name=data_set)

    p.show_grid()
    p.show_axes_all()
    p.show()


def draw_voxels(block_model: BlockModel, data_set_name: str, min_value: float):
    data_set = block_model.get_data_set(data_set_name)
    volume = vedo.Volume(data_set)
    lego = volume.legosurface(vmin=min_value)
    lego.cmap('jet', vmin=min_value, vmax=2*min_value)

    lego.show(axes=1)


def grid(block_model: BlockModel, data_set: str):
    values = block_model.get_data_set(data_set)
    structure = block_model.structure
    grid = pv.UniformGrid()

    grid.dimensions = np.array(values.shape) + 1

    # References
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (
        structure.block_size[0], structure.block_size[1], structure.block_size[2])

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

    (x, y, z) = mgrid[start_x:end_x:delta_x,
                      start_y:end_y:delta_y, start_z:end_z:delta_z]
    grid = pv.StructuredGrid(x, y, z)

    grid['vol'] = values.flatten(order='F')
    grid.contour([0.5, 0.7, 1.2]).plot()


def draw_footprint(footprint: Footprint, blockmodel: BlockModel, values_2d: np.ndarray = np.array([])):
    structure = blockmodel.structure

    footprint_indices = footprint.footprint_indices
    footprint_heights = footprint.footprint_height

    number_of_items = len(footprint_indices[footprint_indices > 0])  # Numbers
    vertices = np.zeros([number_of_items * __vertices_per_cube, 3])
    faces = np.zeros([number_of_items * __faces_per_cube, 3]).astype('int')
    values = np.zeros([number_of_items * __scalars_per_cube])

    if values_2d.size == 0:
        values_2d = footprint_heights

    current_index = -1
    for i in np.arange(structure.shape[0]):
        for j in np.arange(structure.shape[1]):
            if (footprint_indices[i, j] > 0):

                current_index = current_index + 1

                height = footprint_heights[i, j]
                value = values_2d[i, j]
                (current_vertices, current_faces) = __get_cube(
                    (i, j), structure, height, current_index)

                vertices[current_index * __vertices_per_cube:(
                    current_index+1) * __vertices_per_cube, :] = current_vertices
                faces[current_index *
                      __faces_per_cube:(current_index+1) * __faces_per_cube, :] = current_faces

                values[current_index *
                       __scalars_per_cube:(current_index+1) * __scalars_per_cube] = value
    footprint_mesh = pv.make_tri_mesh(vertices, faces)
    footprint_mesh['values'] = values
    footprint_mesh.plot(show_axes=True, show_bounds=True, cmap='jet')
    return
    surf = pv.PolyData(vertices, faces)

    # surf['values'] = values

    # plot each face with a different color
    # scalars=np.arange(3)
    surf.plot()
    print('hola')


def draw_columns_from_arrays(x_values: np.ndarray, y_values: np.ndarray, floors: np.ndarray,
                             heights: np.ndarray, x_size: float, y_size: float,
                             benefit: np.ndarray, slopes_percentage: np.ndarray):
    number_of_items = len(x_values)  # Numbers
    vertices = np.zeros([number_of_items * __vertices_per_cube, 3])
    faces = np.zeros([number_of_items * __faces_per_cube, 3]).astype('int')
    benefit_scalars = np.zeros([number_of_items * __scalars_per_cube])
    height_scalars = np.zeros([number_of_items * __scalars_per_cube])
    floor_scalars = np.zeros([number_of_items * __scalars_per_cube])
    slope_percentage_scalars = np.zeros([number_of_items * __scalars_per_cube])
    slope_angle_scalars = np.zeros([number_of_items * __scalars_per_cube])

    benefit_name = 'Beneficio MUSD'
    height_name = 'Altura columna m'
    level_name = 'Cota m'
    slope_percentage_name = 'Pendiente %'
    slope_angle_name = 'Pendiente º'

    selected_scalar = slope_angle_name

    for current_index in np.arange(number_of_items):
        height = heights[current_index]
        floor = floors[current_index]
        value = benefit[current_index]
        x = x_values[current_index]
        y = y_values[current_index]
        slope_percentage = slopes_percentage[current_index]

        (current_vertices, current_faces) = __get_cube(x,
                                                       y, floor, height, x_size, y_size, current_index)

        vertices[current_index * __vertices_per_cube:(
            current_index+1) * __vertices_per_cube, :] = current_vertices
        faces[current_index *
              __faces_per_cube:(current_index+1) * __faces_per_cube, :] = current_faces
        benefit_scalars[current_index *
                        __scalars_per_cube:(current_index+1) * __scalars_per_cube] = value
        height_scalars[current_index *
                       __scalars_per_cube:(current_index+1) * __scalars_per_cube] = height
        floor_scalars[current_index *
                      __scalars_per_cube:(current_index+1) * __scalars_per_cube] = floor

        slope_percentage_scalars[current_index *
                                 __scalars_per_cube:(current_index+1) * __scalars_per_cube] = slope_percentage

        slope_angle_scalars[current_index *
                            __scalars_per_cube:(current_index+1) * __scalars_per_cube] = np.arctan(slope_percentage/100)*180/np.pi

    footprint_mesh = pv.make_tri_mesh(vertices, faces)
    footprint_mesh[benefit_name] = benefit_scalars/1e6
    footprint_mesh[height_name] = height_scalars
    footprint_mesh[level_name] = floor_scalars
    footprint_mesh[slope_percentage_name] = slope_percentage_scalars
    footprint_mesh[slope_angle_name] = slope_angle_scalars

    footprint_mesh.set_active_scalars(selected_scalar)

    my_theme = themes.DocumentTheme()

    font_size = 12
    my_theme.font.size = font_size
    my_theme.cmap = 'jet'
    my_theme.colorbar_horizontal.height = 0.04
    my_theme.font.label_size = font_size
    my_theme.font.title_size = font_size + 4
    my_theme.title = ''

    my_theme.axes.box = True

    pv.set_plot_theme(my_theme)
    plotter = pv.Plotter()

    clim = [0, 0]
    if (footprint_mesh.active_scalars_name == benefit_name):
        clim = [5e-2, 1]
    elif (footprint_mesh.active_scalars_name == height_name):
        clim = [height_scalars.min(), height_scalars.max()]
    elif (footprint_mesh.active_scalars_name == level_name):
        clim = [floor_scalars.min(), floor_scalars.max()]
    elif (footprint_mesh.active_scalars_name == slope_percentage_name):
        clim = [0, 50]
    elif (footprint_mesh.active_scalars_name == slope_angle_name):
        clim = [0, 50]
    else:
        raise NameError("Out of available names")

    cmap = ['#0000A6', '#00A800', '#FF2121']
    plotter.add_mesh(footprint_mesh, smooth_shading=True,
                     split_sharp_edges=True, cmap=cmap, clim=clim, below_color='gray', above_color='#A600A6')

    def set_xy_plane(status: bool):
        plotter.camera_position = 'xy'

    plotter.add_checkbox_button_widget(set_xy_plane)
    plotter.show_grid()
    plotter.camera_position = 'xy'
    plotter.enable_parallel_projection()
    plotter.show()

    return plotter


def __get_cube(subscripts: Tuple[int, int], structure: BlockModelStructure, height: float, current_index: int) -> Tuple[np.ndarray, np.ndarray]:
    block_size = structure.block_size
    offset = structure.offset
    factor = __scale_factor
    vertices = np.array([[-.5, -.5, -.5], [+.5, -.5, -.5], [+.5, +.5, -.5], [-.5, +.5, -.5],
                        [-.5, -.5, +.5], [+.5, -.5, +.5], [+.5, +.5, +.5], [-.5, +.5, +.5]])

    vertices[:, 0] = vertices[:, 0] * block_size[0] * factor + \
        subscripts[0] * block_size[0] - block_size[0]/2 + offset[0]
    vertices[:, 1] = vertices[:, 1] * block_size[1] * factor + \
        subscripts[1] * block_size[1] - block_size[1]/2 + offset[1]
    vertices[:, 2] = vertices[:, 2] * height + height/2 + offset[2]

    faces = np.array([[5, 4, 0], [1, 5, 0], [6, 5, 1], [2, 6, 1], [7, 6, 2], [3, 7, 2], [
                     4, 7, 3], [0, 4, 3], [6, 7, 4], [5, 6, 4], [1, 0, 3], [2, 1, 3]]).astype('int')
    faces = faces + current_index * __vertices_per_cube
    return (vertices, faces)


def __get_cube(x: float, y: float, floor: float, height: float, x_size: float, y_size: float, current_index: int) -> Tuple[np.ndarray, np.ndarray]:

    factor = __scale_factor
    vertices = np.array([[-.5, -.5, -.5], [+.5, -.5, -.5], [+.5, +.5, -.5], [-.5, +.5, -.5],
                        [-.5, -.5, +.5], [+.5, -.5, +.5], [+.5, +.5, +.5], [-.5, +.5, +.5]])

    vertices[:, 0] = vertices[:, 0] * x_size + x
    vertices[:, 1] = vertices[:, 1] * y_size + y
    vertices[:, 2] = vertices[:, 2] * height + height/2 + floor

    faces = np.array([[5, 4, 0], [1, 5, 0], [6, 5, 1], [2, 6, 1], [7, 6, 2], [3, 7, 2], [
                     4, 7, 3], [0, 4, 3], [6, 7, 4], [5, 6, 4], [1, 0, 3], [2, 1, 3]]).astype('int')
    faces = faces + current_index * __vertices_per_cube
    return (vertices, faces)
