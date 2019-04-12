from DatasetsConsumers.SensitiveEnron import SensitiveEnron

DATA_PATH = "data/SensitiveEnron/evidence_tampering"


class EnronEvidence(SensitiveEnron):
    def common_load(self, data_path, load_filtered_data):
        return super().common_load(data_path, load_filtered_data)

    def load(self, load_filtered_data=False):
        emails, labels = self.common_load(DATA_PATH, load_filtered_data)
        super().post_load(emails, labels)
        return emails, labels
