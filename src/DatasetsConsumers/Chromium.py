from DatasetsConsumers.CommonDevConsumer import CommonDevConsumer


class Chromium(CommonDevConsumer):
    def common_load(self, json_path, mails_path, load_filtered_data):
        return super().common_load(json_path, mails_path, load_filtered_data)

    def load(self, load_filtered_data=False):
        if load_filtered_data:
            load_check_result = super().pre_load()
            if load_check_result is not None:
                return load_check_result
        words, labels = self.common_load('../../data/Chromium/chromium-dev.json/chromium-dev.json',
                                         '../../data/Chromium/chromium-dev/',
                                         load_filtered_data)
        super().post_load(words, labels)
        return words, labels
