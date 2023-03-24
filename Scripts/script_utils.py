import json

class Column:
    value: float
    level_meters: float
    height_meters: float
    subscript_i: int
    subscript_j: int
    x: float
    y: float

    def __init__(self, value: float, level_meters: float, height_meters: float,
                 subscript_i: int, subscript_j: int, x: float, y: float) -> None:
        self.value = value
        self.level_meters = level_meters
        self.height_meters = height_meters
        self.subscript_i = subscript_i
        self.subscript_j = subscript_j
        self.x = x
        self.y = y


class Columns:
    storage: list[Column]

    def __init__(self) -> None:
        self.storage = []

    def add_column(self, column: Column):
        self.storage.append(column)

    def export_json(self, file_path: str):
        items: list[dict[str, float]] = []

        for column in self.storage:
            items.append({
                "value": column.value,
                "level_meters": column.level_meters,
                "height_meters": column.height_meters,
                "subscript_i": column.subscript_i,
                "subscript_j": column.subscript_j,
                "x": column.x,
                "y": column.y})

        with open(file_path, "w") as write:
            json.dump(items, write)

    def import_json(file_path: str):
        columns = Columns()
        with open(file_path, 'r') as read:
            items: list[dict[str, float]] = json.loads(read.readline())

        for item in items:
            value = item['value']
            level_meters = item['level_meters']
            height_meters = item['height_meters']
            subscript_i = item['subscript_i']
            subscript_j = item['subscript_j']
            x = item['x']
            y = item['y']
            columns.add_column(Column(value, level_meters,
                               height_meters, subscript_i, subscript_j, x, y))
        return columns