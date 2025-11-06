from rapidfuzz.distance import Levenshtein
from typing import List, Dict
from collections import defaultdict

from dataset import get_label
from .base import BaseEvaluator

class QwenVLEvaluator(BaseEvaluator):
    def __init__(self, base_model, lora_path = None, device: str = "auto", **kwargs):
        super().__init__(base_model, lora_path, device, **kwargs)
        self.predict_history = {'ground_truth':[], 'prediction': []}

    @staticmethod
    def _cal_exact(prediction_label, reference_label):
        diff = Levenshtein.distance(prediction_label, reference_label)
        diff = (len(reference_label) - diff) if (len(reference_label) - diff) > 0 else 0
        exact_metric = diff / len(reference_label)
        return exact_metric

    def evaluate_text_similarity(self, reference, prediction):
        #metric = super().evaluate_text_similarity(reference, prediction)
        metric = {}
        reference_label = get_label(self.dataset_name, reference) #.split(':')[0].lower()
        prediction_label = get_label(self.dataset_name, prediction) #.split(':')[0].lower()

        best_metric = 0
        for ref in reference_label:
            for pred in prediction_label:
                cur_metric = QwenVLEvaluator._cal_exact(pred, ref)
                if cur_metric > best_metric:
                    best_metric = cur_metric

        metric['Exact'] = best_metric

        self.predict_history['ground_truth'].append(reference_label[0])
        self.predict_history['prediction'].append(prediction_label[0])

        return metric

    def get_predict_history(self):
        return self.predict_history

    def calculate_metrics(self, results: List[Dict], use_metric = ['Exact']) -> Dict:
        return super().calculate_metrics(results, use_metric)