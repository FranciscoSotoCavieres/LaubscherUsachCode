import numpy as np
from Constants.NumericConstants import MIN_VALUE
from Engine.CavingProductionPlanTarget import CavingProductionPlanTarget
from Engine.ExtractionPeriodBasicScheduleResult import ExtractionPeriodBasicScheduleResult
from Models.Footprint import Footprint
from Models.Sequence import Sequence
from Models.utils import FootprintSubscript
from Constants.NumericConstants import MIN_VALUE


class MaximumExtractionInformation:
    """Maximum extraction informtion per query
    """
    days_of_extraction: float
    out_of_range: bool
    if_extracted_is_depleted: bool
    maximum_tonnage: float

    def __init__(self, days_of_extraction: float, out_of_range: bool,
                 if_extracted_is_depleted: bool, maximum_tonnage: float):
        self.days_of_extraction = days_of_extraction
        self.out_of_range = out_of_range
        self.if_extracted_is_depleted = if_extracted_is_depleted
        self.maximum_tonnage = maximum_tonnage


class ColumnMaximumExtractionEngine:
    _days: list[float]
    _maximum_extraction_tonnage: list[float]
    _last_day: float
    _current_day: float

    def __init__(self, column, target: CavingProductionPlanTarget):
        self._days = []
        self._maximum_extraction_tonnage = []
        self._last_day = []
        self._current_day = 0

        column_tonnage = column.total_tonnage
        column_area = column.area

        self._days.append(0)
        self._maximum_extraction_tonnage.append(0)

        current_mass = 0
        current_day = 0
        for i in np.arange(len(target.speed_items)):
            speed_item = target.speed_items[i]
            extraction_speed = speed_item.extraction_tonnes_per_day_squared_meters
            tonnage = column_tonnage * \
                (speed_item.maximum_percentage -
                 speed_item.minimum_percentage) / 100

            current_day = current_day + tonnage / \
                (extraction_speed * column_area)
            current_mass = current_mass + tonnage
            self._days.append(current_day)
            self._maximum_extraction_tonnage.append(current_mass)

        self._last_day = self._days[len(self._days) - 1]

    def simulate_maximum_extraction(self, days_of_extraction: float) -> MaximumExtractionInformation:
        """Maximum tonnage available

        Args:
            days_of_extraction (float): days in which the column will extract
        Returns:
            MaximumExtractionInformation: _description_
        """

        if (self._current_day + days_of_extraction > self._last_day):
            out_of_range = True
            if_extracted_depleted = True
            days_of_extraction = self._last_day - self._current_day
            maximum_tonnage = self.get_tonnage(
                self._last_day) - self.get_tonnage(self._current_day)
            return MaximumExtractionInformation(days_of_extraction, out_of_range, if_extracted_depleted, maximum_tonnage)
        else:
            actual_tonnage = self.get_tonnage(self._current_day)
            final_tonnage = self.get_tonnage(
                self._current_day + days_of_extraction)
            maximum_tonnage = final_tonnage - actual_tonnage
            return MaximumExtractionInformation(days_of_extraction=days_of_extraction, if_extracted_is_depleted=False,
                                                maximum_tonnage=maximum_tonnage, out_of_range=False)

    def update_current_day(self, tonnage: float):
        current_tonnage = self.get_tonnage(self._current_day)
        self._current_day = self.get_day(tonnage + current_tonnage)

    def get_day(self, tonnage: float) -> float:
        """Get interpolated day from tonnage

        Args:
            tonnage (float): tonnage to interpolate

        Returns:
            float: interpolated day
        """
        index = ColumnMaximumExtractionEngine._index_of(
            tonnage, self._maximum_extraction_tonnage)
        ColumnMaximumExtractionEngine._index_raise_error(index)
        xp = [self._maximum_extraction_tonnage[index],
              self._maximum_extraction_tonnage[index+1]]
        yp = [self._days[index], self._days[index+1]]
        day = np.interp(tonnage, xp=xp, fp=yp)
        return day

    def get_tonnage(self, day: float) -> float:
        """Get the interpolated tonnage from a day

        Args:
            day (float): day to interpolate

        Returns:
            float: interpolated tonnage
        """
        index = ColumnMaximumExtractionEngine._index_of(day, self._days)
        ColumnMaximumExtractionEngine._index_raise_error(index)
        xp = [self._days[index], self._days[index+1]]
        yp = [self._maximum_extraction_tonnage[index],
              self._maximum_extraction_tonnage[index+1]]
        tonnage = np.interp(day, xp=xp, fp=yp)
        return tonnage

    @staticmethod
    def _index_raise_error(index: int):
        if (index == -1):
            raise Exception("Invalid index")

    @staticmethod
    def _index_of(value: float, array: list[float]) -> int:
        for i in np.arange(len(array)-1):
            if (array[i] - MIN_VALUE <= value and value <= array[i+1] + MIN_VALUE):
                return i
        return -1


class ProductionPlanColumn:
    # Column State
    total_tonnage: float
    available_tonnage: float
    current_meters: float
    column_height_meters: float
    is_depleted: bool
    depleted_period: int
    available_tonnage_vector: np.ndarray
    is_activated: bool
    activated_period: int

    # Column metadata
    subscript: FootprintSubscript
    sequence_index: int
    tonnage_vector: np.ndarray
    block_height: float
    area: float

    # Column Engine
    engine: ColumnMaximumExtractionEngine

    # results
    results: list[ExtractionPeriodBasicScheduleResult]

    def __init__(self, footprint: Footprint, subscript: FootprintSubscript, sequence: Sequence, density: np.ndarray, target: CavingProductionPlanTarget):

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

        # Engine
        self.engine = ColumnMaximumExtractionEngine(self, target)

    def simulate_maximum_extraction(self, days_of_extraction: float) -> MaximumExtractionInformation:
        return self.engine.simulate_maximum_extraction(days_of_extraction)

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
                                                                    self.is_depleted, self.subscript)
            self.results.append(extraction_result)
            self.depleted_period = period_id
            self.engine.update_current_day(extraction_result.extracted_tonnage)
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
                                                                    self.is_depleted, self.subscript)
            self.results.append(extraction_result)
            self.depleted_period = period_id
            self.engine.update_current_day(extraction_result.extracted_tonnage)
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
                to_meters = to_meters + (target_tonnage /  self.tonnage_vector[i]) * self.block_height
                target_tonnage = 0
                self.current_meters = to_meters
            # Extract the whole block
            else:
                self.available_tonnage = self.available_tonnage - block_tonnage
                to_meters =  self.block_height * (i+1)
                self.available_tonnage_vector[i] = 0
                target_tonnage = target_tonnage - block_tonnage
                self.current_meters = to_meters

        extraction_result = ExtractionPeriodBasicScheduleResult(original_target_tonnage, self.available_tonnage,
                                                                original_target_tonnage, True, from_meters, to_meters,
                                                                period_id, self.is_depleted, self.subscript)
        self.results.append(extraction_result)

        self.engine.update_current_day(extraction_result.extracted_tonnage)
        return extraction_result

    def activate_column(self, period_id: int):
        self.is_activated = True
        self.activated_period = period_id
