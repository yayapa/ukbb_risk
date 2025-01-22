import os

import pandas as pd
from pycox.evaluation import EvalSurv
import torch
import numpy as np


class ModelEvaluator:
    def __init__(self, model, test_data):
        self.model = model
        self.test_x = test_data[0]
        self.test_t = test_data[1][0]
        self.test_e = test_data[1][1]

    def evaluate(self, compute_baseline_hazards=False):
        if compute_baseline_hazards:
            _ = self.model.compute_baseline_hazards()
        preds = self.model.predict_surv_df(self.test_x)
        test_durations_np = (
            self.test_t.numpy()
            if isinstance(self.test_t, torch.Tensor)
            else self.test_t
        )
        test_events_np = (
            self.test_e.numpy()
            if isinstance(self.test_e, torch.Tensor)
            else self.test_e
        )
        evaluator = EvalSurv(preds, test_durations_np, test_events_np, censor_surv="km")
        self.evaluator = evaluator
        self.c_index = evaluator.concordance_td("antolini")
        # integrated brier score
        quantiles = np.linspace(0, 1, num=25)  # 25 quantiles from min to max
        times = np.quantile(test_durations_np, quantiles)
        self.ibs = evaluator.integrated_brier_score(times)

        print(f"Concordance Index: {self.c_index}")
        print(f"IBS: {self.ibs:.4f}")
        return evaluator

    def save_results(self, path):
        metrics = {"c_index": self.c_index, "ibs": self.ibs}
        metrics = pd.DataFrame(metrics, index=[0])
        metrics.to_csv(os.path.join(path, "metrics.csv"), index=False)
