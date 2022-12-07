import numpy as np
from Models.BlockModelStructure import BlockModelStructure

structure_keyword = 'STRUCTURE_RESERVED'


class BlockModel:
    structure: BlockModelStructure
    dataset: dict[str, np.ndarray]

    def __init__(self, structure: BlockModelStructure):
        self.structure = structure
        self.dataset = dict([])

    def add_dataset(self, name: str, data: np.ndarray):
        # Check for dimensions
        if not self.structure.check_dimensions(data):
            raise Exception('Dimension error')
        # Check for name existence
        if name in self.get_dataset_names():
            raise Exception(f'{name} already exists')
        self.dataset[name] = data

    def update_dataset(self, name: str, data: np.ndarray):
        # Check for dimensions
        if self.structure.check_dimensions(data):
            raise Exception('Dimension error')
        if name not in self.get_dataset_names():
            raise Exception(f'{name} doesn''t already exists')
        # Update element
        self.dataset[name] = data

    def get_dataset_names(self) -> list[str]:
        return list(self.dataset.keys())

    def get_data_set(self, dataset: str) -> np.ndarray:
        return self.dataset[dataset]

    def delete_data_set(self, dataset: str):
        self.dataset.pop(dataset)

    def exists_data_set(self, name: str):
        if name in self.get_dataset_names():
            return True
        return False

    def clone(self):
        old_structure = self.structure
        new_structure = BlockModelStructure(block_size=old_structure.block_size.copy(),
                                            shape=old_structure.shape.copy(), offset=old_structure.offset.copy())
        new_block_model = BlockModel(new_structure)
        for data_set_name in self.get_dataset_names():
            new_block_model.add_dataset(
                data_set_name, self.get_data_set(data_set_name).copy())
        return new_block_model

    def save(self, filepath: str):
        data_set_dict: dict[str, np.ndarray] = dict([])
        for data_set_name in self.get_dataset_names():
            data_set_dict[data_set_name] = self.get_data_set(data_set_name)
        structure_array = np.array(
            [self.structure.shape, self.structure.offset, self.structure.block_size])
        data_set_dict[structure_keyword] = structure_array
        np.save(filepath, data_set_dict)

    def export_csv(self, filepath: str):
        header = "x,y,z"
        
        lines :list[str] = []        

        data_set_dict: dict[str, np.ndarray] = dict()
        structure = self.structure

        for data_set_name in self.get_dataset_names():
            data_set_dict[data_set_name] = self.get_data_set(data_set_name)
            header = f"{header},{data_set_name}"
        header = f"{header}\n"
        lines.append(header)

        for i in np.arange(structure.shape[0]):
            for j in np.arange(structure.shape[1]):
                for k in np.arange(structure.shape[2]):
                    center = structure.get_centroid(i, j, k)
                    line = f"{center[0]},{center[1]},{center[2]}"
                    for data_set_name in self.get_dataset_names():
                            value = data_set_dict[data_set_name][i,j,k]
                            line = f"{line},{value}"
                    line = f"{line}\n"
                    lines.append(line)
                    
        
        with open(filepath,"w+") as file:
            file.writelines(lines)
