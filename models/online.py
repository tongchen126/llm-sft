import json
from openai import OpenAI
import base64
from tqdm import tqdm
import time
import json
from tqdm import tqdm
import numpy as np
from typing import List, Dict
from rapidfuzz.distance import Levenshtein

from .base import BaseEvaluator
from dataset import get_label
from utils import gpt_api

class OnlineEvaluator(BaseEvaluator):
    def __init__(self, base_model, dataset_name = None, **kwargs):
        self.model = base_model
        self.predict_history = {'ground_truth':[], 'prediction': []}
        self.dataset_name = dataset_name
    def generate_response(
        self, processed, max_length: int = 2048, temperature=0.7, top_p=0.8
    ) -> tuple:
        # Extract information from processed item
        # Assuming processed contains conversation messages and image_path
        system_message = None
        user_message = None
        image_path = None
    
        for message in processed:
            if message['role'] == 'system':
                # Extract text from system content
                for content_item in message['content']:
                    if content_item['type'] == 'text':
                        system_message = content_item['text']
                        break
            
            elif message['role'] == 'user':
                # Extract text and image from user content
                for content_item in message['content']:
                    if content_item['type'] == 'text':
                        user_message = content_item['text']
                    elif content_item['type'] == 'image':
                        image_path = content_item['url']
    
    # Generate response using gpt_api
        response = gpt_api(
            model=self.model,
            system=system_message,
            user=user_message,
            image_path=image_path
        )

        return system_message + user_message, response

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
                cur_metric = OnlineEvaluator._cal_exact(pred, ref)
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