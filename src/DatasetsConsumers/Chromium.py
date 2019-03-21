from DatasetsConsumers.CommonDevConsumer import CommonDevConsumer

JSON_PATH = '../../../data/Chromium/chromium-dev.json/chromium-dev.json'
DATA_PATH = '../../../data/Chromium/chromium-dev/'


class Chromium(CommonDevConsumer):
    def common_load(self, json_path, mails_path, load_filtered_data):
        return super().common_load(json_path, mails_path, load_filtered_data)

    def load(self, load_filtered_data=False):
        if load_filtered_data:
            load_check_result = super().pre_load()
            if load_check_result is not None:
                return load_check_result
        words, labels = self.common_load(JSON_PATH, DATA_PATH, load_filtered_data)
        super().post_load(words, labels)
        return words, labels

    def n_categories(self):
        super().n_categories()
