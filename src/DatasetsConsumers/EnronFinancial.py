from DatasetsConsumers.SensitiveEnron import SensitiveEnron

DATA_PATH = "data/SensitiveEnron/financial_state"


class EnronFinancial(SensitiveEnron):
    def set_classes(self):
        self.classes = ['Sensitive', 'Non-Sensitive']

    def common_load(self, data_path):
        return super().common_load(data_path)

    def sub_load(self):
        return self.common_load(DATA_PATH)
