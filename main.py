# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numbers

import numpy as np

from Models.BlockModelStructure import decimate_meshes, decimate_polygons
from Models.BlockModelStructure import read_polygons
from Models.BlockModelStructure import histogram
import Models.BlockModelStructure as bm
import Models.Factory as ft
from Models.utils import reduce_csv_file_variables

if __name__ == '__main__':
    pass
    filepath = r'C:\Users\franc\bmining\Antonio Rabajille - Carpeta_compartida\Ejercicio Pampa Escondida SLC 2022-01-01\PampaASC.csv'
    out_filepath_csv = r'C:\Users\franc\bmining\Antonio Rabajille - Carpeta_compartida\Ejercicio Pampa Escondida SLC 2022-01-01\PampaASC_reduced.csv'
    out_filepath_python = r'C:\Users\franc\bmining\Antonio Rabajille - Carpeta_compartida\Ejercicio Pampa Escondida SLC 2022-01-01\PampaPython.npy'
    # Units
    x_label = 'xcentre'
    y_label = 'ycentre'
    z_label = 'zcentre'
    cut_label = 'cut'
    density_label = 'densidad'
    mo_label = 'mo'
    codmat = 'codmat'
    categ_rec = 'categ_rec'
    litologia = 'litologia'
    minzone = 'minzone'
    # TODO: Filtrar información

    labels = [x_label, y_label, z_label, cut_label, density_label, mo_label, codmat, categ_rec, litologia, minzone]

    reduce_csv_file_variables(filepath, out_filepath_csv, labels)
    # block_model = ft.block_model_from_csv_file(filepath, x_label, y_label, z_label, ',', labels)
    # block_model.save(out_filepath_python)
    # print(block_model.structure.shape)

    # decimate_polygons()
    # list_of_list_vec3 = statistics_of_lines(r'C:\Users\franc\bmining\Gabriela Alarcón - BHP\Visualización 3D\topographyLines.csv')
