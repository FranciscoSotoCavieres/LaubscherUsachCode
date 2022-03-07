import numpy as np
from Constants.NumericConstants import MIN_VALUE
from Engine.ExtractionPeriodBasicScheduleResult import ExtractionPeriodBasicScheduleResult
from Models.Footprint import Footprint
from Models.Sequence import Sequence
from Models.utils import FootprintSubscript


class ProductionPlanColumn:
    # Column State
    total_tonnage: float
    available_tonnage: float
    current_meters: float
    column_height_meters: float
    is_depleted: bool
    depleted_period:int
    available_tonnage_vector: np.ndarray
    is_activated : bool
    activated_period : int
    
    # Column metadata
    subscript: FootprintSubscript
    sequence_index: int
    tonnage_vector: np.ndarray
    block_height: float
    area : float

    # results
    results: list[ExtractionPeriodBasicScheduleResult]

    def __init__(self, footprint: Footprint, subscript: FootprintSubscript, sequence: Sequence, density: np.ndarray):

        structure = footprint.structure
        block_volume = structure.get_block_volume()
        block_height = structure.block_size[2]
        column_block_height = footprint.footprint_indices[subscript.i, subscript.j]

        # Metadata
        self.subscript = subscript
        self.sequence_index = sequence.sequence_indices[subscript.i, subscript.j]
        self.tonnage_vector = density[subscript.i,
                                      subscript.j, :column_block_height] * block_volume
        self.block_height = block_height
        self.area = structure.get_block_area()

        # State
        self.total_tonnage = np.sum(self.tonnage_vector)
        self.available_tonnage = self.total_tonnage
        self.current_meters = 0
        self.column_height_meters = block_height * column_block_height
        self.available_tonnage_vector = np.copy(self.tonnage_vector)
        
        self.activated_period = -1
        self.is_activated = False
        self.depleted_period = -1
        self.is_depleted = False

        
        # Results
        self.results = []

    def extract(self, target_tonnage: float, period_id: int) -> ExtractionPeriodBasicScheduleResult:
        tonnage_available = self.available_tonnage
        # Case 0: Deplete
        if np.abs(tonnage_available - target_tonnage) < MIN_VALUE:
            self.is_depleted = True
            self.available_tonnage = 0
            self.available_tonnage_vector[:] = 0
            from_meters = self.current_meters
            to_meters = self.column_height_meters
            self.current_meters = self.column_height_meters
            extraction_result = ExtractionPeriodBasicScheduleResult(target_tonnage, 0, target_tonnage, True,
                                                                    from_meters, to_meters, period_id,
                                                                    self.is_depleted,self.subscript)
            self.results.append(extraction_result)
            self.depleted_period = period_id
            return extraction_result
        # Case 1: No enough mass
        if (tonnage_available < target_tonnage):
            self.is_depleted = True
            self.available_tonnage = 0
            self.available_tonnage_vector[:] = 0
            from_meters = self.current_meters
            to_meters = self.column_height_meters
            self.current_meters = self.column_height_meters
            extraction_result = ExtractionPeriodBasicScheduleResult(target_tonnage, 0, tonnage_available, False,
                                                                    from_meters, to_meters, period_id,
                                                                    self.is_depleted,self.subscript)
            self.results.append(extraction_result)
            self.depleted_period = period_id
            return extraction_result

        # Case 2: Enough mass
        from_index = int(self.current_meters/self.block_height)
        to_index = int(self.column_height_meters/self.block_height)
        original_target_tonnage = target_tonnage
        from_meters = self.current_meters
        to_meters = self.current_meters
        for i in np.arange(from_index, to_index):
            block_tonnage = self.available_tonnage_vector[i]
            # Extract part of the block
            if (block_tonnage >= target_tonnage):
                self.available_tonnage = self.available_tonnage - target_tonnage               
    
                self.available_tonnage_vector[i] = self.available_tonnage_vector[i] - target_tonnage
                to_meters = to_meters + (target_tonnage/ self.tonnage_vector[i]) * self.block_height                
                target_tonnage = 0
                self.current_meters = to_meters            
            # Extract the whole block
            else:
                self.available_tonnage = self.available_tonnage - block_tonnage
                to_meters = to_meters + self.block_height
                self.available_tonnage_vector[i] = 0  
                target_tonnage = target_tonnage - block_tonnage
                self.current_meters = to_meters
                
        extraction_result = ExtractionPeriodBasicScheduleResult(original_target_tonnage, self.available_tonnage,
                                                                        original_target_tonnage, True, from_meters, to_meters,
                                                                        period_id,self.is_depleted,self.subscript)
        self.results.append(extraction_result)
        return extraction_result

        
    
    def activate_column(self,period_id:int):
        self.is_activated = True
        self.activated_period = period_id
