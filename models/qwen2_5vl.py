from .base import BaseEvaluator
from rapidfuzz.distance import Levenshtein
from typing import List, Dict

from dataset import get_label

class QwenVLEvaluator(BaseEvaluator):
    def __init__(self, base_model, lora_path = None, device: str = "auto", **kwargs):
        super().__init__(base_model, lora_path, device, **kwargs)

    def evaluate_text_similarity(self, reference, prediction):
        metric = super().evaluate_text_similarity(reference, prediction)
        reference_label = get_label(self.dataset_name, reference) #.split(':')[0].lower()
        prediction_label = get_label(self.dataset_name, prediction) #.split(':')[0].lower()
        diff = Levenshtein.distance(prediction_label, reference_label)
        diff = (len(reference_label) - diff) if (len(reference_label) - diff) > 0 else 0
        exact_metric = diff / len(reference_label)
        metric['Exact'] = exact_metric
        return metric

    def calculate_metrics(self, results: List[Dict], use_metric = ['Semantic-Similarity', 'ROUGE-2-F', 'BLEU-2', 'METEOR', 'Exact']) -> Dict:
        return super().calculate_metrics(results, use_metric)

class QwenVLCOTEvaluator(QwenVLEvaluator):
    def __init__(self, base_model, lora_path = None, device: str = "auto", **kwargs):
        super().__init__(base_model, lora_path, device, **kwargs)
        
    def evaluate_text_similarity(self, reference, prediction):
        metric = super().evaluate_text_similarity(reference, prediction)
        return metric

    def calculate_metrics(self, results: List[Dict], use_metric = ['Semantic-Similarity', 'ROUGE-2-F', 'BLEU-2', 'METEOR', 'Exact']) -> Dict:
        return super().calculate_metrics(results, use_metric)

