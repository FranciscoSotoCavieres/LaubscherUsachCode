from Models.BlockModelDilution import BlockModelDilution
from Models.utils import best_height_of_draw, sequence_footprint
import Models.Factory as ft
import Models.visualization as vs
from Scripts.ValorizationScript import valorization
from Engine.CavingProductionPlanExtractionSpeedItem import CavingProductionPlanExtractionSpeedItem
from Engine.CavingProductionPlanTargetItem import CavingProductionPlanTargetItem
from Engine.CavingProductionPlanTarget import CavingProductionPlanTarget
from Engine.ProductionPlanEngine import ProductionPlanEngine
from Scripts.GualcamayoOptimizer import build_inclined_footprints, compute_tonnage_grade, draw_grades, compute_best_height_level, compute_best_height, generate_plan, segment_squares, stats_footprints
import numpy as np
if __name__ == '__main__':

    print("Hello world")
    exit()
    # Importar modelo de bloques
    mb_path_csv = r'C:\Users\franc\bmining\BM22-AUS33-IUG GUALCAMAYO - Documentos\03 Equipo de Trabajo\09 Francisco Soto\md_gual5x5x5_nov22.csv'
    mb_path_numpy = r'C:\Users\franc\bmining\BM22-AUS33-IUG GUALCAMAYO - Documentos\03 Equipo de Trabajo\09 Francisco Soto\Análisis Caving\md_gual5x5x5_nov22.npy'
    mb_diluted_folder = r'C:\Users\franc\bmining\BM22-AUS33-IUG GUALCAMAYO - Documentos\03 Equipo de Trabajo\09 Francisco Soto\Análisis Caving\Dilution'

    block_model = ft.block_model_from_npy_file(mb_path_numpy)

    au = block_model.get_data_set("AU_FINAL")
    ore = block_model.get_data_set("ORE")
    density = block_model.get_data_set("Density")

    real_au = au * ore

    logic = real_au > 0.15

    mean_average = np.sum(
        real_au[logic] * density[logic]) / (np.sum(density[logic]))
    tonnage = np.sum(density[logic]) * 5 * 5 * 5

    # generate_plan(mb_diluted_folder)
    # build_inclined_footprints(mb_diluted_folder)
    # segment_squares(mb_diluted_folder)
    # stats_footprints(mb_diluted_folder)
    exit()
    block_model = ft.block_model_from_csv_file(path, "X", "Y", "Z")
    data_sets = block_model.get_dataset_names()

    vs.blockmodel_view(block_model, "Cut")

    hola = 1

    # output_block_model_path = r"C:\Users\franc\OneDrive - bmining\Repositorio USACH\Modelos de Bloques\Modelos Csv\G12.npy"
    # block_model.save(output_block_model_path)
    # block_model = ft.block_model_from_npy_file(output_block_model_path)

    # level_block_model = ft.block_model_from_level(
    #     block_model, 7)  # Cortar de modelo de bloques

    # output_block_model_path = r"C:\Users\franc\OneDrive - bmining\Repositorio USACH\Modelos de Bloques\Modelos Csv\G12.npy"

    # # Crear una calculadora de dilución
    # block_dilution = BlockModelDilution(40, block_model.structure)
    # block_dilution.compute_dilution_coefficients()

    # # Diluyen los set de datos asociados.
    # block_model_diluted = level_block_model.clone()
    # for data_set_name in block_model.get_dataset_names():
    #     data_set = block_model.get_data_set(data_set_name)
    #     diluted_data_set = block_dilution.dilute_dataset(data_set)
    #     block_model_diluted.update_dataset(data_set_name, diluted_data_set)

    # valorization(block_model_diluted)
    # path_to_excel_footprint = r'C:\Users\franc\Desktop\Clase Usach 03-06-2022\Footprint1.xlsx'
    # footprint = ft.footprint_from_excel(
    #     path_to_excel_footprint, block_model_diluted)
    # sequence = sequence_footprint(footprint, 30)

    # path_to_excel_sequence = r'C:\Users\franc\Desktop\Clase Usach 03-06-2022\Sequence1.xlsx'
    # sequence.export_to_excel(path_to_excel_sequence)
    # same_sequence = ft.sequence_from_excel(
    #     path_to_excel_sequence, block_model_diluted)

    # # Target
    # first_item_target = CavingProductionPlanTargetItem(1, 1e6, 10, 360)
    # target_items: list[CavingProductionPlanTargetItem] = [first_item_target]

    # # Speed

    # first_item_speed = CavingProductionPlanExtractionSpeedItem(0, 30, 0.15)
    # speed_items: list[CavingProductionPlanExtractionSpeedItem] = [
    #     first_item_speed]

    # # Caving plan
    # path_to_excel_caving_plan_configuration = r'C:\Users\franc\Desktop\Clase Usach 03-06-2022\Config.xlsx'
    # configuration = ft.caving_configuration_from_excel(
    #     path_to_excel_caving_plan_configuration)
    # configuration.denisty_data_set_name = "Density"
    # productionPlanEngine = ProductionPlanEngine(block_model_diluted, footprint,
    #                                             sequence, configuration)
    # result = productionPlanEngine.process()
    # path_to_excel_caving_plan_result = r'C:\Users\franc\Desktop\Clase Usach 03-06-2022\Plan.xlsx'
    # result.export_excel(path_to_excel_caving_plan_result)
    # hola = 1
