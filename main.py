from Models.BlockModelDilution import BlockModelDilution
import Models.Factory as ft
import Models.visualization as vs
from Scripts.ValorizationScript import valorization

if __name__ == '__main__':
    # Importar modelo de bloques
    path = r"C:\Users\franc\OneDrive - bmining\Repositorio USACH\Modelos de Bloques\Modelos Csv\G12.csv"
    block_model = ft.block_model_from_csv_file(path, "X", "Y", "Z")
    output_block_model_path = r"C:\Users\franc\OneDrive - bmining\Repositorio USACH\Modelos de Bloques\Modelos Csv\G12.npy"
    block_model.save(output_block_model_path)
    block_model = ft.block_model_from_npy_file(output_block_model_path)

    level_block_model = ft.block_model_from_level(block_model,7)
    hola = 1
    # output_block_model_path = r"C:\Users\franc\OneDrive - bmining\Repositorio USACH\Modelos de Bloques\Modelos Csv\G12.npy"
    # block_model.save(output_block_model_path)
    # block_model = ft.block_model_from_npy_file(output_block_model_path)

    # block_model_from_level = block_model.clone()
    # block_dilution = BlockModelDilution(40, block_model.structure)
    # block_dilution.compute_dilution_coefficients()

    # block_model_diluted = block_model.clone()
    # for data_set_name in block_model.get_dataset_names():
       # data_set = block_model.get_data_set(data_set_name)
       # diluted_data_set = block_dilution.dilute_dataset(data_set)
       # block_model_diluted.update_dataset(data_set_name, diluted_data_set)

    # valorization(block_model_diluted)
    
