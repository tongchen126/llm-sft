import json
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm
import numpy as np
from typing import List, Dict
from pathlib import Path
from torch.utils.data import Dataset
from peft import PeftModel

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class BaseEvaluator:
    def __init__(self, base_model, lora_path = None, device: str = "auto", base_model_class = AutoModelForImageTextToText):
        self.device = device
        self.model_path = base_model
        self.lora_path = lora_path

        # Load tokenizer and model
        print(f"Loading base model from {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            trust_remote_code=True
        )
        
        self.model = base_model_class.from_pretrained(
            base_model,
            device_map=device,
            trust_remote_code=True,
            dtype=torch.float16,  # Use float16 for memory efficiency
        ).eval()

        if lora_path is not None:
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.processor = AutoProcessor.from_pretrained(base_model)

        print("Model loaded successfully!")
    
    def generate_response(self, processed,max_length: int = 512, temperature = 0.7, top_p = 0.8) -> str:
        input_length = processed['input_ids'].shape[1]
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **processed.to(self.model.device),
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # TODO: Compare the above code to "model.generate(**processed_chat.to(model.device), max_new_tokens=128)"
        # Decode response
        prompt = self.tokenizer.decode(outputs[0][:input_length], skip_special_tokens=True)
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Remove input from response
        
        return prompt, response
    
    def evaluate_dataset(self, dataset: List[Dict], save_results: bool = True, output_file: str = "evaluation_results.json") -> Dict:
        results = []
        
        print(f"Evaluating on {len(dataset)} samples...")
        
        for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                # Extract conversation and image path
                processed, original, ground_truth = item
                # Generate prediction
                prompt, predicted_response = self.generate_response(processed)
                
                result = {
                    'prompt': prompt,
                    'ground_truth': ground_truth,
                    'predicted': predicted_response,
                }
                results.append(result)
                print(f"Prompt:\t{prompt}\nResponse:\t{predicted_response}\n\nGround Truth:\t{ground_truth}\n")
            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue
        
        # Save results
        if save_results:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {output_file}")
        
        # Calculate basic statistics
        stats = {
            "total_samples": len(results),
            "successful_generations": len([r for r in results if r["predicted"]]),
            "average_prediction_length": np.mean([len(r["predicted"]) for r in results if r["predicted"]]),
            "average_ground_truth_length": np.mean([len(r["ground_truth"]) for r in results if r["ground_truth"]])
        }
        
        return {"results": results, "statistics": stats}

    def evaluate_text_similarity(self, reference, prediction):
        """
        Evaluate text similarity using multiple metrics
        """
        results = {}
        
        # 1. BLEU Score (0-1, higher is better)
        # Commonly used for machine translation, measures n-gram overlap
        reference_tokens = reference.lower().split()
        prediction_tokens = prediction.lower().split()
        smoothing = SmoothingFunction().method1
        
        bleu1 = sentence_bleu([reference_tokens], prediction_tokens, 
                            weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu2 = sentence_bleu([reference_tokens], prediction_tokens, 
                            weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu4 = sentence_bleu([reference_tokens], prediction_tokens, 
                            weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        results['BLEU-1'] = bleu1
        results['BLEU-2'] = bleu2
        results['BLEU-4'] = bleu4
        
        # 2. METEOR Score (0-1, higher is better)
        # Considers synonyms and stemming, more flexible than BLEU
        meteor = meteor_score([reference_tokens], prediction_tokens)
        results['METEOR'] = meteor
        
        # 3. ROUGE Score (0-1, higher is better)
        # Measures recall-oriented overlap, commonly used for summarization
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, prediction)
        
        results['ROUGE-1-F'] = rouge_scores['rouge1'].fmeasure
        results['ROUGE-2-F'] = rouge_scores['rouge2'].fmeasure
        results['ROUGE-L-F'] = rouge_scores['rougeL'].fmeasure
        
        # 4. BERTScore (0-1, higher is better)
        # Uses contextual embeddings, captures semantic similarity better
        P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=False)
        results['BERTScore-F1'] = F1.item()
        results['BERTScore-Precision'] = P.item()
        results['BERTScore-Recall'] = R.item()
        
        # 5. Semantic Similarity using Sentence Transformers (0-1, higher is better)
        # Direct semantic comparison using pre-trained models
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([reference, prediction])
        semantic_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        results['Semantic-Similarity'] = semantic_sim
        
        return results

    def calculate_metrics(self, results: List[Dict], use_metric = ['Semantic-Similarity', 'ROUGE-2-F', 'BLEU-2', 'METEOR']) -> Dict:
        calculated_metrics = {key: [] for key in use_metric}
        for result in tqdm(results):
            pred = result['predicted']
            gt = result['ground_truth']
            metric = self.evaluate_text_similarity(gt, pred)
            for key in use_metric:
                calculated_metrics[key].append(metric[key])

        return calculated_metrics
