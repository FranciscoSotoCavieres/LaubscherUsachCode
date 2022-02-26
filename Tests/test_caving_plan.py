import unittest
from matplotlib import pyplot as plt
import numpy as np
from sympy import frac
from Engine.CavingProductionPlanExtractionSpeedItem import CavingProductionPlanExtractionSpeedItem
from Engine.CavingProductionPlanTarget import CavingProductionPlanTarget
import os
import Models.Factory as ft
from Models.BlockModel import BlockModel
from Engine.CavingProductionPlanTargetItem import CavingProductionPlanTargetItem
from Engine.ProductionPlanColumn import ProductionPlanColumn
from Engine.ExtractionPeriodBasicScheduleResult import ExtractionPeriodBasicScheduleResult
from Models.Footprint import Footprint
from Models.Sequence import Sequence
from Models.utils import FootprintSubscript
import pytest


class CavingPlanShould(unittest.TestCase):
    def test_caving_target(self):
        plan_name = 'Panel A'
        density_data_set_name = "Density"

        # Target
        first_item_target = CavingProductionPlanTargetItem(1, 1e6, 10, 360)
        second_item_target = CavingProductionPlanTargetItem(2, 2e6, 20, 360)

        target_items: list[CavingProductionPlanTargetItem] = [
            second_item_target, first_item_target]

        # Speed
        first_item_speed = CavingProductionPlanExtractionSpeedItem(30, 70, 0.2)
        second_item_speed = CavingProductionPlanExtractionSpeedItem(
            0, 30, 0.15)

        speed_items: list[CavingProductionPlanExtractionSpeedItem] = [
            first_item_speed, second_item_speed]

        caving_production_plan_target = CavingProductionPlanTarget(
            plan_name, density_data_set_name, target_items, speed_items)

        filepath = f'{os.getcwd()}/test_data/plan_configuration.xlsx'
        caving_production_plan_target.export_to_excel(filepath)

        caving_production_plan_target_imported = ft.caving_configuration_from_excel(
            filepath)

        assert caving_production_plan_target.target_items[0].period_number == 1
        assert caving_production_plan_target.name == caving_production_plan_target_imported.name
        assert caving_production_plan_target.target_items[
            1].target_tonnage == caving_production_plan_target_imported.target_items[1].target_tonnage
        assert caving_production_plan_target_imported.speed_items[1].maximum_percentage == 70
        assert caving_production_plan_target.denisty_data_set_name == density_data_set_name

    def test_column_unit(self):
        
       
        
        block_model: BlockModel = ft.block_model_from_npy_file(
            f'{os.getcwd()}/test_data/G8Fixed.npy')
        footprint: Footprint = ft.footprint_from_excel(
            f'{os.getcwd()}/test_data/FootprintFixed.xlsx', block_model)
        sequence = ft.sequence_from_excel(
            f'{os.getcwd()}/test_data/SequenceFixed.xlsx', block_model)

        density = block_model.get_data_set('Density')

        footprint_subscript = FootprintSubscript(15, 8)
        value = footprint.footprint_indices[footprint_subscript.i,
                                            footprint_subscript.j]

        assert value * footprint.structure.block_size[2] == 234

        block_volume = footprint.structure.get_block_volume()
        block_height = footprint.structure.block_size[2]

        # Extract the whole column
        production_plan_column = ProductionPlanColumn(
            footprint, footprint_subscript, sequence, density)
        target_tonnage = production_plan_column.available_tonnage
        result = production_plan_column.Extract(target_tonnage, 1)
        assert result.extracted_tonnage == target_tonnage

        # Extract half a block and then another half
        fraction = 0.6
        production_plan_column = ProductionPlanColumn(
            footprint, footprint_subscript, sequence, density)
        target_tonnage = density[footprint_subscript.i,
                                 footprint_subscript.j, 0] * block_volume * fraction
        result = production_plan_column.Extract(target_tonnage, 2)
        assert result.extracted_tonnage == target_tonnage
        assert pytest.approx(
            production_plan_column.current_meters, 0.01) == block_height * fraction
        assert pytest.approx(production_plan_column.total_tonnage -
                             production_plan_column.available_tonnage, 0.01) == target_tonnage

        fraction = 0.4
        target_tonnage = density[footprint_subscript.i,
                                 footprint_subscript.j, 0] * block_volume * fraction
        result = production_plan_column.Extract(target_tonnage, 3)
        assert result.extracted_tonnage == target_tonnage
        assert pytest.approx(
            production_plan_column.current_meters, 0.01) == block_height * 1
        assert pytest.approx(production_plan_column.total_tonnage -
                             production_plan_column.available_tonnage, 0.01) == density[footprint_subscript.i,
                                                                                        footprint_subscript.j, 0] * block_volume
        # Extract the whole column
        available_tonnage = production_plan_column.available_tonnage
        result = production_plan_column.Extract(1e10, 2)
        assert result.was_target_accomplished == False
        assert result.tonnage_available == 0
        assert result.is_depleted == True
        assert pytest.approx(result.extracted_tonnage,
                             0.01) == pytest.approx(available_tonnage, 0.01)
        assert result.tonnage_available == 0
