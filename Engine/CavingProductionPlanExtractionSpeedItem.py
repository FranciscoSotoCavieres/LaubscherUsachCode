class CavingProductionPlanExtractionSpeedItem:
    minimum_percentage : float
    maximum_percentage : float
    extraction_tonnes_per_day_squared_meters:float    
    
    def __init__(self,minimum_percentage: float,maximum_percentage:float,extraction_tonnes_per_day_squared_meters : float):
        self.minimum_percentage = minimum_percentage
        self.maximum_percentage = maximum_percentage
        self.extraction_tonnes_per_day_squared_meters = extraction_tonnes_per_day_squared_meters
        