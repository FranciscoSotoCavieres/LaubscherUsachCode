from typing import List, Sequence
from Engine.CavingProductionPlanTarget import CavingProductionPlanTarget
from Models.BlockModel import BlockModel
from Models.Footprint import Footprint

class CavingProductionPlanEngine:
    blockmodel:  BlockModel
    footprint:  Footprint
    sequence:  Sequence
    target:CavingProductionPlanTarget

    def __init__(self, blockmodel: BlockModel, footprint: Footprint, sequence: Sequence, target: CavingProductionPlanTarget):
        self.blockmodel = blockmodel
        self.footprint = footprint
        self.sequence = sequence
        self.target = target
    
    
