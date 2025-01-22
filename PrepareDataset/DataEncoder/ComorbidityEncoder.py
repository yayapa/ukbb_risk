from PrepareDataset.DataEncoder.BaseDataEncoder import BaseDataEncoder

# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from PrepareDataset.ComorbidityExtractor import ComorbidityExtractor
from PrepareDataset.ICDExtractor import ICDExtractor
import pandas as pd


class ComorbidityEncoder(BaseDataEncoder):
    def __init__(
        self,
        comorbidity_file_path=None,
        icd_code_dict_file_path=None,
        interested_date=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if (
            comorbidity_file_path is None
            and icd_code_dict_file_path is None
            and interested_date is None
        ):
            logger = kwargs["logger"]
            logger.info(
                "comorbidity_file_path, icd_code_dict_file_path and interested_date are not provided"
            )
            return

        comorbidity_extractor = ComorbidityExtractor(comorbidity_file_path)
        self.icd_extractor = ICDExtractor(
            icd_code_dict_file_path=icd_code_dict_file_path
        )
        icd_code_dict = self.icd_extractor.load_icd_code_dict()
        eids_to_read = self.df["eid"].tolist()
        # take only the keys from the dictionary that are in the eids_to_read
        self.icd_code_dict = {
            k: v for k, v in icd_code_dict.items() if k in eids_to_read
        }
        self.comorbidity_dict = comorbidity_extractor.load_comorbidity_data()
        self.interested_date = interested_date

    def encode(self):
        self.df = pd.DataFrame(columns=(["eid"] + list(self.comorbidity_dict.keys())))
        # fill the dataframe with eids form icd_code_dict and 0s in other columns
        for eid in self.icd_code_dict.keys():
            row = [eid]
            for comorbidity_class in self.comorbidity_dict.keys():
                icd_selector = self.comorbidity_dict[comorbidity_class]
                data = self.icd_extractor.get_all_before_date(
                    eid, self.icd_code_dict[eid][self.interested_date]
                )
                if any(icd in icd_selector for icd in data["icd_codes"]):
                    row.append(1)
                else:
                    row.append(0)
            self.df.loc[len(self.df)] = row

        return self.df
