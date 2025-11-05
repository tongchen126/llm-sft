import os
import torch
import argparse
from torch.utils.data import Subset
import json
import numpy as np
from collections import defaultdict

from models import QwenVLEvaluator
from dataset import load_dataset
from utils import *

def main(model_path, dataset_path, image_base_path, load_json = None, output_file = "evaluation_results.json", dataset_name = 'pokemon'):
    parser = argparse.ArgumentParser(description="Evaluate Qwen-VL model on ShareGPT dataset")
    parser.add_argument("--model_path", type=str, default=model_path, help="Path to base model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to lora")
    parser.add_argument("--dataset_path", type=str, default=dataset_path, help="Path to ShareGPT JSON dataset")
    parser.add_argument("--image_base_path", type=str, default=image_base_path, help="Base path for images")
    parser.add_argument("--output_file", type=str, default=output_file, help="Output file for results")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to evaluate")
    parser.add_argument("--load_json", type=str, default=load_json, help="Load json file")

    args = parser.parse_args()
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
    model = 'qwen3vl-30bA3b'
    save_dir = f'tmp/{model}-r1/'
    dataset_name = ('pokemon', 'pokemon1')
    dataset_json = ['train_eval', 'eval']
    data_dir = f'/workspace/user_code/workspace_40172/llm-sft/data/{dataset_name[1]}'
    dir_dict = list_directories(os.path.join(f"../LLaMA-Factory/saved/{dataset_name[1]}/", model))
    output_metrics = defaultdict(dict)
    excluded_keys = [] # ["sft-7-e4-full-b1", "sft-8-e5-full-b1"]
    os.makedirs(save_dir, exist_ok = True)
    mode = 'generate'
    for key, val in dir_dict.items():
        if key in excluded_keys:
            continue
        if mode == 'generate':
            for data_name in dataset_json:
                main(val, os.path.join(data_dir, f'data_{data_name}.json'), data_dir, output_file=os.path.join(save_dir, model + '-' + key + f'_{data_name}.json'),
                    dataset_name=dataset_name[0])
        elif mode == 'eval':
            for data_name in dataset_json:
                metrics, predict_history = main(val, os.path.join(data_dir, f'data_{data_name}.json'), data_dir,
                                            load_json=os.path.join(save_dir, model + '-' + key + f'_{data_name}.json'), dataset_name=dataset_name[0])
                output_metrics[data_name][key] = metrics
                fig = plot_prediction_heatmap(predict_history['ground_truth'], predict_history['prediction'], annot=False)
                fig.savefig(os.path.join(save_dir, model + '-' + key + f'_{data_name}.png'))

    if mode == 'eval':
        plot_losses_from_json([os.path.join(val, 'trainer_state.json') for key, val in dir_dict.items()],
                                list(dir_dict.keys()), os.path.join(save_dir, model + '_loss.png'))
        for key, val in output_metrics:
            df = save_model_performance_table(val, os.path.join(save_dir, model + f'_{key}_perf.html'), format='html')
