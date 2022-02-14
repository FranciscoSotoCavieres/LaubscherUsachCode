from typing import List, Sequence
from Models.BlockModel import BlockModel
from Models.Footprint import Footprint


class CavingProductionPlanEngine:
    blockmodel:  BlockModel
    footprint:  Footprint
    sequence:  Sequence

    def __init__(self, blockmodel: BlockModel, footprint: Footprint, sequence: Sequence):
        self.blockmodel = blockmodel
        self.footprint = footprint



    


    
