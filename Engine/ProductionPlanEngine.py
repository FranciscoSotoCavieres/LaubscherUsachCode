from Engine.CavingProductionPlanTarget import CavingProductionPlanTarget
from Engine.ExtractionPeriodBasicScheduleResult import ExtractionPeriodBasicScheduleResult
from Engine.ProductionPlanColumn import MaximumExtractionInformation, ProductionPlanColumn
from Engine.ProductionPlanResult import ProductionPlanResult
from Models.BlockModel import BlockModel
from Models.BlockModelStructure import BlockModelStructure
from Models.Footprint import Footprint
from Models.Sequence import Sequence
import numpy as np
from Models.utils import FootprintSubscript


class ProductionPlanEngine:
    blockmodel:  BlockModel
    footprint:  Footprint
    sequence:  Sequence
    target: CavingProductionPlanTarget
    structure: BlockModelStructure
    columns: np.ndarray

    acceleration_ratio = 0.9
    """ Ratio of a column when is required
    """

    def __init__(self, blockmodel: BlockModel, footprint: Footprint, sequence: Sequence, target: CavingProductionPlanTarget):
        self.blockmodel = blockmodel
        self.footprint = footprint
        self.sequence = sequence
        self.target = target
        self.structure = self.blockmodel.structure

        self.columns = np.empty(
            [self.structure.shape[0], self.structure.shape[1]], dtype=ProductionPlanColumn)

        density = blockmodel.get_data_set(self.target. denisty_data_set_name)
        for i in np.arange(self.structure.shape[0]):
            for j in np.arange(self.structure.shape[1]):
                if footprint.footprint_indices[i, j] <= 0:
                    continue
                column: ProductionPlanColumn = ProductionPlanColumn(
                    footprint, FootprintSubscript(i, j), sequence, density, target)
                self.columns[i, j] = column

    def process(self) -> ProductionPlanResult:
        results: list[ExtractionPeriodBasicScheduleResult] = []
        target_items = self.target.target_items

        # Gets available columns and sort by sequence index
        columns_pool = self.columns[np.where(self.columns != None)]
        columns_pool = sorted(
            columns_pool, key=lambda x: x.sequence_index)

        current_days = 0
        for target_item in target_items:
            period_id = target_item.period_number

            # Active new columns
            to_incorporate = target_item.incorporation_blocks
            for column in columns_pool:
                if (to_incorporate == 0):
                    break
                if (column.is_activated):
                    continue
                column.activate_column(period_id)
                to_incorporate = to_incorporate - 1

            # -- Maximum Extraction --
            # Get the maximum extraction available
            subscripts_maximum_extraction_result: dict[FootprintSubscript,
                                                       MaximumExtractionInformation] = dict()
            # Get the active and not depleted column
            columns_available: np.ndarray = np.array(list(filter(
                lambda x: x.is_activated == True and x.is_depleted == False, columns_pool)))

            # Check if exists available columns
            if len(columns_available) == 0:
                continue

            days_of_extraction = target_item.duration_days
            target_tonnage = target_item.target_tonnage

            for index in np.arange(len(columns_available)):
                column: ProductionPlanColumn = columns_available[index]

                subscripts = column.subscript

                maximum_extraction_result = column.simulate_maximum_extraction(
                    days_of_extraction)

                subscripts_maximum_extraction_result[subscripts] = maximum_extraction_result

            # -- Extraction process --
            maximum_tonnage = sum(
                [x.maximum_tonnage for x in subscripts_maximum_extraction_result.values()])

            if (maximum_tonnage == 0):
                ratio = 0
            else:
                ratio = target_tonnage/maximum_tonnage
            if (ratio <= 1):
                period_results = self._extract_period(period=period_id,
                                                      available_columns=columns_available,
                                                      ratio=ratio,
                                                      subscripts_maximum_extraction_result=subscripts_maximum_extraction_result,
                                                      tonnage=target_tonnage)
                for period_result in period_results:
                    results.append(period_result)
            else:
                period_results = self._maximum_extraction(columns=columns_available,
                                                          subscripts_maximum_extraction_result=subscripts_maximum_extraction_result,
                                                          period=period_id)
                for period_result in period_results:
                    results.append(period_result)

            # Upgrade variables
            current_days = current_days + days_of_extraction

        average_sets = self.target.average_data_sets
        summation_sets = self.target.summation_data_sets
        production_plan_result = ProductionPlanResult(
            units=results, block_model=self.blockmodel, target=self.target, average_sets=average_sets, summation_sets=summation_sets)
        return production_plan_result

    def _maximum_extraction(self, period: int, columns: np.ndarray, subscripts_maximum_extraction_result: dict[FootprintSubscript, MaximumExtractionInformation]) -> list[ExtractionPeriodBasicScheduleResult]:
        extraction_results:  list[ExtractionPeriodBasicScheduleResult] = []

        for index in np.arange(len(columns)):
            column: ProductionPlanColumn = columns[index]
            maximum_extraction = subscripts_maximum_extraction_result[
                column.subscript].maximum_tonnage
            result = column.extract(
                period_id=period, target_tonnage=maximum_extraction)
            extraction_results.append(result)
        return extraction_results

    def _extract_period(self, tonnage: float, period: int, ratio: float, available_columns: np.ndarray,
                        subscripts_maximum_extraction_result: dict[FootprintSubscript, MaximumExtractionInformation]) -> list[ExtractionPeriodBasicScheduleResult]:

        valid_columns = sorted(available_columns, key=lambda x: 1 -
                               x.available_tonnage/x.total_tonnage)

        extraction_results:  list[ExtractionPeriodBasicScheduleResult] = []

        current_tonnage_to_extract = tonnage

        last_index = 0
        # --  Extract the accelaration ratio --
        for index in np.arange(len(valid_columns)):
            column: ProductionPlanColumn = valid_columns[index]
            if (1 - column.available_tonnage/column.total_tonnage < self.acceleration_ratio):
                break  # No more acceleration items
            subscripts = column.subscript

            maximum_extraction_result = subscripts_maximum_extraction_result[subscripts]

            # Case of not enough mass -- Extract everything available
            if (maximum_extraction_result.maximum_tonnage < current_tonnage_to_extract):
                maximum_tonnage = maximum_extraction_result.maximum_tonnage
                extraction_result = column.extract(period_id=period,
                                                   target_tonnage=maximum_tonnage)
                extraction_results.append(extraction_result)

                current_tonnage_to_extract = current_tonnage_to_extract - \
                    extraction_result.extracted_tonnage
            else:
                extraction_result = column.extract(
                    target_tonnage=current_tonnage_to_extract, period_id=period)
                extraction_results.append(extraction_result)
                current_tonnage_to_extract = 0
                break

            last_index = index + 1

        # Case of not more tonnage
        if current_tonnage_to_extract <= 0:
            return extraction_results

        # -- Normal extraction --
        for index in np.arange(last_index, len(available_columns), 1):
            column: ProductionPlanColumn = valid_columns[index]
            maximum_extraction_result = subscripts_maximum_extraction_result[column.subscript]
            to_extract = maximum_extraction_result.maximum_tonnage * ratio
            result = column.extract(
                target_tonnage=to_extract, period_id=period)
            extraction_results.append(result)

        return extraction_results
