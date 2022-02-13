import csv
import os
import open3d as o3d
import numpy as np
from ezdxf.math import Vec3
from rdp import rdp
from numpy import ndarray
from open3d.cpu.pybind.geometry import SimplificationContraction, TriangleMesh, LineSet
from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector


class BlockModelStructure:
    block_size: ndarray
    offset: ndarray
    shape: np.ndarray

    def __init__(self, block_size: np.ndarray, shape: np.ndarray, offset: np.ndarray):
        self.offset = offset
        self.shape = shape
        self.block_size = block_size

    def get_block_volume(self) -> float:
        return self.block_size[0] * self.block_size[1] * self.block_size[2]

    def get_block_area(self) -> float:
        return self.block_size[0] * self.block_size[1]

    def check_dimensions(self, data: np.ndarray):
        if len(data.shape) != 3:
            return False
        if data.shape[0] == self.block_size[0] \
                & data.shape[1] == self.block_size[1] \
                & data.shape[2] == self.block_size[2]:
            return True
        return False

    def get_subscripts(self, x: float, y: float, z: float):
        i = self.get_subscript(x, 'i')
        j = self.get_subscript(y, 'j')
        k = self.get_subscript(z, 'k')
        return np.array([i, j, k], dtype=int)

    def get_centroid(self, subscript_i: int, subscript_j: int, subscript_k: int):
        x = subscript_i * self.block_size[0] + self.offset[0]
        y = subscript_j * self.block_size[1] + self.offset[1]
        z = subscript_k * self.block_size[2] + self.offset[2]

        return np.array([x, y, z])

    def get_left_down_near_corner(self):
        x = - self.block_size[0] / 2 + self.offset[0]
        y = - self.block_size[1] / 2 + self.offset[1]
        z = - self.block_size[2] / 2 + self.offset[2]
        return np.array([x,y,z])

    def get_subscript(self, value: float, subscript_type='i') -> int:
        index = 0
        if subscript_type == 'j':
            index = 1
        elif subscript_type == 'k':
            index = 2

        init = self.offset[index]
        block_size = self.block_size[index]
        size = self.shape[index]

        subscript = (value - init) / block_size
        if subscript >= size:
            return int(subscript - 1)
        if subscript < 0:
            return 0
        return int(subscript)

    def get_data_set_from_1D(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, datasets: dict[str, np.ndarray]):

        indices = np.zeros([x.shape[0], 3], dtype=int)
        for i in range(x.shape[0]):
            indices[i, :] = self.get_subscripts(x[i], y[i], z[i])

        output: dict[str, np.ndarray] = dict([])
        for dataset_name in datasets.keys():
            data_1d = datasets[dataset_name]
            array = np.zeros(self.shape)
            for i in range(x.shape[0]):
                array[indices[i, 0], indices[i, 1], indices[i, 2]] = data_1d[i]
            output[dataset_name] = array
        return output

    @staticmethod
    def from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray):
        unique_x: np.ndarray = np.unique(x)
        unique_y: np.ndarray = np.unique(y)
        unique_z: np.ndarray = np.unique(z)

        unique_x.sort()
        unique_y.sort()
        unique_z.sort()

        block_size_x: float = unique_x[1] - unique_x[0]
        block_size_y: float = unique_y[1] - unique_y[0]
        block_size_z: float = unique_z[1] - unique_z[0]

        offset_x: float = unique_x[0]
        offset_y: float = unique_y[0]
        offset_z: float = unique_z[0]

        shape_x: int = unique_x.shape[0]
        shape_y: int = unique_y.shape[0]
        shape_z: int = unique_z.shape[0]

        return BlockModelStructure(np.array([block_size_x, block_size_y, block_size_z]),
                                   np.array([shape_x, shape_y, shape_z]),
                                   np.array([offset_x, offset_y, offset_z]))


def visualize(mesh: TriangleMesh, newMesh: TriangleMesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    line_set: LineSet = o3d.geometry.LineSet.create_from_triangle_mesh(mesh=newMesh)
    line_set.paint_uniform_color([1, 0, 0])
    newMesh.paint_uniform_color([1, 0, 0])

    # vis.add_geometry(o3d.geometry.LineSet.create_from_triangle_mesh(mesh=mesh))
    vis.add_geometry(mesh)
    vis.add_geometry(newMesh)

    vis.run()
    vis.destroy_window()

    # print(os.listdir(directory))
    # visualize(mesh, new_mesh)


def write_mesh_vertex_file(filePath: str, mesh: TriangleMesh):
    file = open(filePath, 'w', newline='')
    writer = csv.writer(file)
    for triangle in mesh.triangles:
        writer.writerow(mesh.vertices[triangle[0]])
        writer.writerow(mesh.vertices[triangle[1]])
        writer.writerow(mesh.vertices[triangle[2]])
    file.close()


def write_vector3_vertex_file(filePath: str, vertices: [Vec3]):
    file = open(filePath, 'w', newline='')
    writer = csv.writer(file)

    for vertex in vertices:
        writer.writerow(vertex)
    file.close()


def read_dxf_vertices_fast(dxf_path: str):
    face_name = "AcDbFace"

    vertices: list[Vec3] = []
    with open(dxf_path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    for i in range(len(lines)):
        current_line: str = lines[i]
        if current_line == face_name:
            x1 = float(lines[i + 2])
            y1 = float(lines[i + 4])
            z1 = float(lines[i + 6])

            x2 = float(lines[i + 8])
            y2 = float(lines[i + 10])
            z2 = float(lines[i + 12])

            x3 = float(lines[i + 14])
            y3 = float(lines[i + 16])
            z3 = float(lines[i + 18])
            vertices.append(Vec3(x1, y1, z1))
            vertices.append(Vec3(x2, y2, z2))
            vertices.append(Vec3(x3, y3, z3))
    return vertices


def get_mesh(vec3s: [Vec3]) -> TriangleMesh:
    vertices: Vector3dVector = Vector3dVector()
    triangles: Vector3iVector = Vector3iVector()
    currentTriangle = 0

    for index in range(0, len(vec3s), 3):
        vertices.append(vec3s[index])
        vertices.append(vec3s[index + 1])
        vertices.append(vec3s[index + 2])
        triangles.append([currentTriangle, currentTriangle + 1, currentTriangle + 2])
        currentTriangle += 3
    return TriangleMesh(vertices, triangles)


def read_polygons(filepath: str):
    with open(filepath) as file:
        lines = file.readlines()

    list_of_list_vec3: [[Vec3]] = []

    for line in lines:
        list_of_vec3: [Vec3] = []
        for vec3str in line.split(';'):
            vec3_splits = vec3str.split(',')
            vec3 = Vec3(float(vec3_splits[0]), float(vec3_splits[1]), float(vec3_splits[2]))
            list_of_vec3.append(vec3)
        list_of_list_vec3.append(list_of_vec3)
    return list_of_list_vec3


def histogram(list_of_list_vec3: list[list[Vec3]]):
    count: list[int] = []
    for list_of_vec3 in list_of_list_vec3:
        count.append(len(list_of_vec3))
    with open(r'C:\Users\franc\bmining\Gabriela Alarcón - BHP\Visualización 3D\countOfLines.csv', 'w') as file:
        write = csv.writer(file)
        write.writerows([count])


def reduce_polygons(list_of_list_vec3: list[list[Vec3]]) -> list[np.ndarray]:
    list_of_vertex_array: list[np.ndarray()] = []
    minimum_vertices = 50
    for list_vec3 in list_of_list_vec3:
        if len(list_vec3) < minimum_vertices:
            continue
        array = transform_to_array(list_vec3)
        list_of_vertex_array.append(rdp(array, 5))
    return list_of_vertex_array


def get_escondida_vertices(list_of_vertices: list[np.ndarray]):
    out_list_of_vertices: list[np.ndarray] = []
    for vertices in list_of_vertices:
        if points_in_escondida(vertices):
            out_list_of_vertices.append(vertices)
    return out_list_of_vertices


def get_escondida_norte_vertices(list_of_vertices: list[np.ndarray]):
    out_list_of_vertices: list[np.ndarray] = []
    for vertices in list_of_vertices:
        if points_in_escondida_norte(vertices):
            out_list_of_vertices.append(vertices)
    return out_list_of_vertices


def write_csv_polygons(list_array_of_vertices: list[np.ndarray], filepath: str):
    lines: list[str] = []
    for array_of_vertices in list_array_of_vertices:
        vertex_string_list: list[str] = []
        for i in range(array_of_vertices.shape[0]):
            vertex = array_of_vertices[i, :]
            vertex_string_list.append(f'{vertex[0]},{vertex[1]},{vertex[2]};')
        vertex_string = ''.join(vertex_string_list)
        vertex_string = vertex_string[:-1]
        vertex_string += '\n'
        lines.append(vertex_string)
    with open(filepath, 'w') as file:
        file.writelines(lines)


def transform_to_array(list_of_vec3: list[Vec3]):
    array = np.ndarray([len(list_of_vec3), 3])
    currentIter = 0
    for vec3 in list_of_vec3:
        array[currentIter, 0] = vec3.x
        array[currentIter, 1] = vec3.y
        array[currentIter, 2] = vec3.z
        currentIter += 1
    return array


def points_in_escondida_norte(points: np.ndarray) -> bool:
    escondida_norte_rect = np.array([[16876, 112428], [20320, 115552]])
    for i in range(points.shape[0]):
        if point_in_rect(points[i, :], escondida_norte_rect):
            return True
    return False


def points_in_escondida(points: np.ndarray) -> bool:
    escondida_rect = np.array([[14038, 105644], [18074, 110212]])
    for i in range(points.shape[0]):
        if point_in_rect(points[i, :], escondida_rect):
            return True
    return False


def point_in_rect(point: np.ndarray, rect: np.ndarray):
    x1 = rect[0, 0]
    y1 = rect[0, 1]
    x2 = rect[1, 0]
    y2 = rect[1, 1]

    x = point[0]
    y = point[1]
    if x1 < x < x2:
        if y1 < y < y2:
            return True
    return False


def decimate_meshes():
    input_directory = r'C:\Users\franc\bmining\Gabriela Alarcón - BHP\Visualización 3D\Archivos Conformetrics\\'
    output_directory = r'C:\Users\franc\bmining\Gabriela Alarcón - BHP\Visualización 3D\F11_FM_Sportfire_Diciembre_dxf\Archivos Salida 2\vertices folder'

    for filename in os.listdir(input_directory):
        if filename.endswith(".dxf"):
            vertices = read_dxf_vertices_fast(os.path.join(input_directory, filename))
            csvPath = os.path.join(output_directory, filename)
            pre, ext = os.path.splitext(csvPath)
            csvPath = pre + ".csv"

            mesh = get_mesh(vertices)
            mesh = mesh.simplify_vertex_clustering(10, SimplificationContraction.Average)
            write_mesh_vertex_file(csvPath, mesh)


def decimate_polygons():
    list_of_list_vec3 = read_polygons(
        r'C:\Users\franc\bmining\Gabriela Alarcón - BHP\Visualización 3D\topographyLines.csv')
    reduced_polygons = reduce_polygons(list_of_list_vec3)

    escondida_polygons = get_escondida_vertices(reduced_polygons)
    escondida_norte_polygons = get_escondida_norte_vertices(reduced_polygons)

    write_csv_polygons(escondida_polygons,
                       r'C:\Users\franc\bmining\Gabriela Alarcón - BHP\Visualización 3D\escondidaReducedLines.csv')
    write_csv_polygons(escondida_norte_polygons,
                       r'C:\Users\franc\bmining\Gabriela Alarcón - BHP\Visualización 3D\escondidaNorteReducedLines.csv')
