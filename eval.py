import os
import torch
import argparse
from torch.utils.data import Subset

from models import QwenVLEvaluator

def main(model_path, dataset_path):
    parser = argparse.ArgumentParser(description="Evaluate Qwen-VL model on ShareGPT dataset")
    parser.add_argument("--model_path", type=str, default=model_path, help="Path to base model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to lora")
    parser.add_argument("--dataset_path", type=str, default=dataset_path, help="Path to ShareGPT JSON dataset")
    parser.add_argument("--image_base_path", type=str, default=dataset_path, help="Base path for images")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = QwenVLEvaluator(args.model_path, lora_path = args.lora_path, device = args.device)
    
    # Load dataset
    dataset = evaluator.load_dataset(args.dataset_path, args.image_base_path)
    
    # Limit samples if specified
    if args.max_samples:
        dataset = Subset(dataset, range(args.max_samples))
    
    # Evaluate
    evaluation_results = evaluator.evaluate_dataset(
        dataset, 
        save_results=True, 
        output_file=args.output_file
    )
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(evaluation_results["results"])
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples processed: {evaluation_results['statistics']['total_samples']}")
    print(f"Successful generations: {evaluation_results['statistics']['successful_generations']}")
    print(f"Average prediction length: {evaluation_results['statistics']['average_prediction_length']:.1f} chars")
    print(f"Average ground truth length: {evaluation_results['statistics']['average_ground_truth_length']:.1f} chars")
    # print(f"Response rate: {metrics['response_rate']:.2%}")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main('/workspace/user_code/workspace/LLaMA-Factory/saved/qwen2_5vl-7b/lora/sft-1-cot/checkpoint-470', '/workspace/user_code/workspace/llm-sft/data/pokemon_cot')
