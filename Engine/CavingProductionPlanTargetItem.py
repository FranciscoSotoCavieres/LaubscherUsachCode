class CavingProductionPlanTargetItem:
    period_number: int
    incorporation_blocks: int
    target_tonnage: float
    duration_days: float

    def __init__(self, period_number: int, target_tonnage: float, incorporation_blocks: int, duration_days: float):
        self.period_number = period_number
        self.target_tonnage = target_tonnage
        self.incorporation_blocks = incorporation_blocks
        self.duration_days = duration_days