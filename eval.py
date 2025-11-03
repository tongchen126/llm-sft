import os
import torch
import argparse
from torch.utils.data import Subset
import json
import numpy as np

from models import QwenVLEvaluator
from dataset import load_dataset
from utils import *

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
    dataset_name = 'pokemon'
    # Initialize evaluator
    evaluator = QwenVLEvaluator(args.model_path, lora_path = args.lora_path, device = args.device, dataset_name = dataset_name)
    
    if args.load_json is None:
        # Load dataset
        dataset = load_dataset(evaluator.processor, args.dataset_path, args.image_base_path,max_samples=args.max_samples, dataset_name = dataset_name)
        
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
            predict_history = evaluator.get_predict_history()
            # print(f"metrics: {metrics}")
            for key in metrics:
                print(f"Metric {key} mean: {np.mean(metrics[key])}")
            return metrics, predict_history

if __name__ == "__main__":
    model = 'qwen3vl-8b'
    data_dir = '/workspace/user_code/workspace_40172/llm-sft/data/pokemon1'
    dir_dict = list_directories(os.path.join("../LLaMA-Factory/saved/", model))
    output_metrics = {}
    excluded_keys = [] # ["sft-7-e4-full-b1", "sft-8-e5-full-b1"]
    mode = 'eval'
    for key, val in dir_dict.items():
        if key in excluded_keys:
            continue
        if mode == 'generate':
            main(val, data_dir, output_file=os.path.join("tmp", model + '-' + key+'.json'))
        elif mode == 'eval':
            metrics, predict_history = main(val, data_dir, load_json=os.path.join("tmp", model + '-' + key+'.json'))
            output_metrics[key] = metrics
            fig = plot_prediction_heatmap(predict_history['ground_truth'], predict_history['prediction'], annot=False)
            fig.savefig(os.path.join("tmp", model + '-' + key+'.png'))

    if mode == 'eval':
        df = save_model_performance_table(output_metrics, os.path.join('tmp', model + '_perf.html'), format='html')
        plot_losses_from_json([os.path.join(val, 'trainer_state.json') for key, val in dir_dict.items()],
                                list(dir_dict.keys()), os.path.join('tmp', model+'_loss.png'))
