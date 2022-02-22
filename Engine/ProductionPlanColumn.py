import math
import numpy as np
from Constants.NumericConstants import MIN_VALUE
from Engine.ExtractionPeriodBasicScheduleResult import ExtractionPeriodBasicScheduleResult


class CavingGradeReserveUnit:
    total_tonnage: float
    available_tonnage: float
    current_meters: float
    column_height: float
    is_depleted: bool
    results: list[ExtractionPeriodBasicScheduleResult]

    # TODO: Colocar los tonelajes array

    def __init__(self) -> None:
        self.total_tonnage = 0
        self.is_depleted = False

    def Extract(self, target_tonnage: float, period_id: int) -> ExtractionPeriodBasicScheduleResult:
        tonnage_available = self.available_tonnage

        extractionPeriodBasicSchedule = ExtractionPeriodBasicScheduleResult()

        # Case 0: Deplete

        if np.abs(tonnage_available - target_tonnage) < MIN_VALUE:
            self.is_depleted = True
            self.available_tonnage = 0

            from_meters = self.current_meters
            to_meters = self.column_height
            self.current_meters = self.column_height
            return ExtractionPeriodBasicScheduleResult(target_tonnage, 0, target_tonnage, True, from_meters, to_meters,
                                                       period_id)
