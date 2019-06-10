from enum import Enum


class DatasetMode(Enum):
    def __str__(self):
        return str(self.value)
    SIZE_FULL = "standard"
    SIZE_2000 = "2000"
