from Engine.CavingProductionPlanTarget import CavingProductionPlanTarget
from Engine.ProductionPlanColumn import ProductionPlanColumn
from Models.BlockModel import BlockModel
from Models.BlockModelStructure import BlockModelStructure
from Models.Footprint import Footprint
from Models.Sequence import Sequence
import numpy as np
from Models.utils import FootprintSubscript


class CavingProductionPlanEngine:
    blockmodel:  BlockModel
    footprint:  Footprint
    sequence:  Sequence
    target: CavingProductionPlanTarget
    structure: BlockModelStructure
    columns: np.ndarray[ProductionPlanColumn]

    def __init__(self, blockmodel: BlockModel, footprint: Footprint, sequence: Sequence, target: CavingProductionPlanTarget):
        self.blockmodel = blockmodel
        self.footprint = footprint
        self.sequence = sequence
        self.target = target
        self.structure = self.blockmodel.structure
        self.columns = np.empty(
            [self.structure.shape[0], self.structure.shape[1]],dtype=ProductionPlanColumn)

        density = blockmodel.get_data_set(self.target. denisty_data_set_name)
        for i in np.arange(self.structure.shape[0]):
            for j in np.arange(self.structure.shape[1]):
                column: ProductionPlanColumn = ProductionPlanColumn(
                    footprint, FootprintSubscript(i, j), sequence, density)
                self.columns[i, j] = column
    
