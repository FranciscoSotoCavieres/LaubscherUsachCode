class ExtractionPeriodBasicScheduleResult:
    target_tonnage: float
    tonnage_available: float
    extracted_tonnage: float
    was_target_accomplished: bool
    from_meters: float
    to_meters: float
    period_id: int

    def __init__(self, target_tonnage: float, tonnage_available: float, extracted_tonnage: float,
                 was_target_accomplished: bool, from_meters:float, to_meters:float,period_id:int) -> None:
        self.target_tonnage = target_tonnage
        self.tonnage_available = tonnage_available
        self.extracted_tonnage = extracted_tonnage
        self.was_target_accomplished = was_target_accomplished
        self.from_meters = from_meters
        self.to_meters = to_meters
        self.period_id = period_id
