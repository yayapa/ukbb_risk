import pandas as pd


class ComorbidityExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.comorbidity_dict = self.load_comorbidity_data()

    def load_comorbidity_data(self):
        comorbidity = pd.read_csv(self.file_path, sep="\t")
        comorbidity_classes = list(
            set(comorbidity.columns.tolist())
            - {"coding", "meaning", "node_id", "parent_id", "selectable"}
        )
        comorbidity_dict = {}
        for comorbidity_class in comorbidity_classes:
            comorbidity_dict[comorbidity_class] = comorbidity[
                comorbidity[comorbidity_class] == 1
            ]["coding"].tolist()

        return comorbidity_dict

    def extract_icd_codes(self, comorbidity_selector):
        icd_selector = []
        for comorbidity_class in comorbidity_selector:
            icd_selector += self.comorbidity_dict[comorbidity_class]
        return icd_selector
