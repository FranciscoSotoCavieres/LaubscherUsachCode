import numpy as np
from Constants import NumericConstants
from Engine.CavingProductionPlanTarget import CavingProductionPlanTarget
from Engine.ProductionPlanColumn import ProductionPlanColumn


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
    _current_day : float

    def __init__(self, column: ProductionPlanColumn, target: CavingProductionPlanTarget):
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

    def get_maximum_extraction(self, days_of_extraction: float) -> MaximumExtractionInformation:
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
            self._current_day  = self._last_day
            return MaximumExtractionInformation(days_of_extraction, out_of_range, if_extracted_depleted, maximum_tonnage)
        else:
            actual_tonnage = self.get_tonnage(self._current_day)
            self._current_day = self._current_day + days_of_extraction
            final_tonnage = self.get_tonnage(self._current_day)
            maximum_tonnage = final_tonnage - actual_tonnage

            return MaximumExtractionInformation(days_of_extraction=days_of_extraction, if_extracted_is_depleted=False,
                                                maximum_tonnage=maximum_tonnage, out_of_range=False)

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
        day = np.interp(tonnage, xp=xp, yp=yp)
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
        tonnage = np.interp(day, xp=xp,fp=yp)
        return tonnage

    @staticmethod
    def _index_raise_error(index: int):
        if (index == -1):
            raise Exception("Invalid index")

    @staticmethod
    def _index_of(value: float, array: list[float]) -> int:
        for i in np.arange(len(array)-1):
            if (array[i] - NumericConstants.MIN_VALUE <= value and value <= array[i+1] + NumericConstants.MIN_VALUE):
                return i
        return -1
