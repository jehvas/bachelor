from DatasetsConsumers.SensitiveEnron import SensitiveEnron

DATA_PATH = "data/SensitiveEnron/evidence_tampering"


class EnronEvidence(SensitiveEnron):
    def set_classes(self) -> None:
        self.classes = ['Sensitive', 'Non-Sensitive']

    def common_load(self, data_path):
        return super().common_load(data_path)

    def sub_load(self):
        return self.common_load(DATA_PATH)
