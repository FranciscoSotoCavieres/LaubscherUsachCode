import unittest

from matplotlib import pyplot as plt
import numpy as np
from Engine.CavingProductionPlanExtractionSpeedItem import CavingProductionPlanExtractionSpeedItem
from Engine.CavingProductionPlanTarget import CavingProductionPlanTarget
import os
import Models.Factory as ft

from Engine.CavingProductionPlanTargetItem import CavingProductionPlanTargetItem


class CavingPlanShould(unittest.TestCase):
    def test_caving_target(self):
        plan_name = 'Panel A'

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
            plan_name, target_items,speed_items)

        filepath = f'{os.getcwd()}/test_data/plan_configuration.xlsx'
        caving_production_plan_target.export_to_excel(filepath)

        caving_production_plan_target_imported = ft.caving_configuration_from_excel(
            filepath)

        assert caving_production_plan_target.target_items[0].period_number == 1
        assert caving_production_plan_target.name == caving_production_plan_target_imported.name
        assert caving_production_plan_target.target_items[
            1].target_tonnage == caving_production_plan_target_imported.target_items[1].target_tonnage
        assert caving_production_plan_target_imported.speed_items[1].maximum_percentage == 70
        
