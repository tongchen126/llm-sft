import os
import torch
import argparse
from torch.utils.data import Subset
import json
import numpy as np

from models import QwenVLEvaluator
from dataset import load_dataset

def main(model_path, dataset_path, load_json = None, output_file = "evaluation_results.json"):
    parser = argparse.ArgumentParser(description="Evaluate Qwen-VL model on ShareGPT dataset")
    parser.add_argument("--model_path", type=str, default=model_path, help="Path to base model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to lora")
    parser.add_argument("--dataset_path", type=str, default=dataset_path, help="Path to ShareGPT JSON dataset")
    parser.add_argument("--image_base_path", type=str, default=dataset_path, help="Base path for images")
    parser.add_argument("--output_file", type=str, default=output_file, help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to evaluate")
    parser.add_argument("--load_json", type=str, default=load_json, help="Load json file")

    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = QwenVLEvaluator(args.model_path, lora_path = args.lora_path, device = args.device)
    
    if args.load_json is None:
        # Load dataset
        dataset = load_dataset(evaluator.processor, args.dataset_path, args.image_base_path,max_samples=args.max_samples)
        
        # Evaluate
        evaluation_results = evaluator.evaluate_dataset(
            dataset, 
            save_results=True, 
            output_file=args.output_file
        )
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total samples processed: {evaluation_results['statistics']['total_samples']}")
        print(f"Successful generations: {evaluation_results['statistics']['successful_generations']}")
        print(f"Average prediction length: {evaluation_results['statistics']['average_prediction_length']:.1f} chars")
        print(f"Average ground truth length: {evaluation_results['statistics']['average_ground_truth_length']:.1f} chars")
        print(f"Results saved to: {args.output_file}")
    else:
        with open(args.load_json, 'r') as f:
            results = json.load(f)
            metrics = evaluator.calculate_metrics(results)
            # print(f"metrics: {metrics}")
            for key in metrics:
                print(f"Metric {key} mean: {np.mean(metrics[key])}")

if __name__ == "__main__":
    main('/workspace/user_code/workspace/LLaMA-Factory/saved/qwen2_5vl-7b/lora/sft-2/', 
        '/workspace/user_code/workspace/llm-sft/data/pokemon')

