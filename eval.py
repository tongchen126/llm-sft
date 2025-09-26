import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any
import argparse
from pathlib import Path
from datasets import Dataset
from torch.utils.data import Dataset

class TAGS:
    def __init__(self,IMAGE_TAG, MESSAGE_TAG, ASSISTANT_TAG, USER_TAG, SYSTEM_TAG, ROLE_TAG, CONTENT_TAG, IMAGE_LABEL=''):
        self.IMAGE_TAG = IMAGE_TAG
        self.MESSAGE_TAG = MESSAGE_TAG
        self.ASSISTANT_TAG = ASSISTANT_TAG
        self.USER_TAG = USER_TAG
        self.SYSTEM_TAG = SYSTEM_TAG
        self.ROLE_TAG = ROLE_TAG
        self.CONTENT_TAG = CONTENT_TAG
        self.IMAGE_LABEL = IMAGE_LABEL

class LLMDataset(Dataset):
    def __init__(self, data_dict, keys):
        """
        Args:
            data_dict (dict): Dictionary with keys 'data' and 'gt'.
        """
        self.data = data_dict
        self.keys = keys
        
    def __len__(self):
        # Return the length of the dataset
        return len(self.data[self.keys[0]])
    
    def __getitem__(self, idx):
        return [self.data[key][idx] for key in self.keys]

class QwenVLEvaluator:
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the Qwen-VL model evaluator
        
        Args:
            model_path: Path to your fine-tuned model
            device: Device to load model on ("auto", "cuda", "cpu")
        """
        self.device = device
        self.model_path = model_path

        # Load tokenizer and model
        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            dtype=torch.float16,  # Use float16 for memory efficiency
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_path)
        # help(self.processor)

        print("Model loaded successfully!")

    def transform_conversation_to_qwen(input_data, TAG, base_url, system_message=None, skip_role = []):
        """
        Transform conversation data from input format to desired output format.
        
        Args:
            input_data (dict): Input conversation data
            base_url (str): Base URL to prepend to image paths
            system_message (str): Optional system message to add at the beginning
        
        Returns:
            list: Transformed conversation in the desired format
        """
        output_messages = []
        
        # Add system message if provided
        if system_message:
            output_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            })
        
        # Process each message
        for message in input_data[TAG.MESSAGE_TAG]:
            role = message[TAG.ROLE_TAG]
            content = message[TAG.CONTENT_TAG]
            if role in skip_role:
                continue
            
            # Parse content and replace <image> placeholders
            content_list = []
            image_index = 0
            
            # Split content by <image> tags and process
            parts = content.split(TAG.IMAGE_LABEL)
            
            for i, part in enumerate(parts):
                # Add text content if not empty
                if part.strip():
                    content_list.append({"type": "text", "text": part.strip()})
                
                # Add image if not the last part (meaning there was an <image> tag after this part)
                if i < len(parts) - 1 and image_index < len(input_data["images"]):
                    image_path = input_data[TAG.IMAGE_TAG][image_index]
                    # Extract filename and create full URL
                    # filename = image_path.split("/")[-1]
                    image_url = str(Path(base_url, image_path))
                    
                    content_list.append({"type": "image", "url": image_url})
                    image_index += 1
            
            # If content_list is empty, add the original content as text
            if not content_list:
                content_list.append({"type": "text", "text": content})
            
            output_messages.append({
                "role": role,
                "content": content_list
            })
        
        return output_messages

    def get_gt_qwen(self, input_data, GT_ROLE, TAG):
        for message in input_data[TAG.MESSAGE_TAG]:
            role = message[TAG.ROLE_TAG]
            content = message[TAG.CONTENT_TAG]
            if role == GT_ROLE:
                return content

    def load_sharegpt_dataset(self, dataset_path: str, image_base_path: str = None, json_name = 'data.json', 
                                transform_func = transform_conversation_to_qwen) -> List[Dict]:
        """
        Load ShareGPT format dataset
        
        Args:
            dataset_path: Path to the ShareGPT JSON file
            image_base_path: Base path where images are stored
            
        Returns:
            List of conversation dictionaries
        """
        TAG = TAGS('images', 'messages', "assistant", "user", "system", "role", "content", "<image>")

        with open(str(Path(dataset_path, json_name)), 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Process dataset to handle image paths
        processed_dataset = {'processed':[],'original': [], 'gt': []}
        for item in dataset[:10]:
            gt = self.get_gt_qwen(item, TAG.ASSISTANT_TAG, TAG)
            original_item = transform_func(item, TAG, image_base_path, system_message = None)

            processed_item = transform_func(item, TAG, image_base_path, system_message = None, skip_role = [TAG.ASSISTANT_TAG])
            processed_chat = self.processor.apply_chat_template(processed_item, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")

            processed_dataset['original'].append(original_item)
            processed_dataset['processed'].append(processed_chat)
            processed_dataset['gt'].append(gt)
            
        processed_dataset = LLMDataset(processed_dataset, ["processed","original","gt"])

        return processed_dataset
    
    def generate_response(self, processed,max_length: int = 512) -> str:
        """
        Generate response from the model
        
        Args:
            input_text: Input text (potentially with image)
            max_length: Maximum generation length
            
        Returns:
            Generated response
        """
        input_length = processed['input_ids'].shape[1]
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **processed.to(self.model.device),
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        prompt = self.tokenizer.decode(outputs[0][:input_length], skip_special_tokens=True)
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Remove input from response
        
        return prompt, response
    
    def evaluate_dataset(self, dataset: List[Dict], save_results: bool = True, output_file: str = "evaluation_results.json") -> Dict:
        """
        Evaluate the model on the entire dataset
        
        Args:
            dataset: ShareGPT format dataset
            save_results: Whether to save results to file
            output_file: Output file path for results
            
        Returns:
            Evaluation results dictionary
        """
        results = []
        
        print(f"Evaluating on {len(dataset)} samples...")
        
        for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                # Extract conversation and image path
                processed, original, ground_truth = item
                # Generate prediction
                prompt, predicted_response = self.generate_response(processed)
                
                results.append(predicted_response)
                print(f"Prompt {prompt}\n Response {predicted_response}\n Ground Truth {ground_truth}\n")
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
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate evaluation metrics (you can extend this with more sophisticated metrics)
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        total_samples = len(results)
        non_empty_predictions = len([r for r in results if r["predicted"].strip()])
        
        metrics = {
            "response_rate": non_empty_predictions / total_samples if total_samples > 0 else 0,
            "average_response_length": np.mean([len(r["predicted"]) for r in results if r["predicted"]]),
            "total_samples": total_samples
        }
        
        return metrics

def main(model_path, dataset_path):
    parser = argparse.ArgumentParser(description="Evaluate Qwen-VL model on ShareGPT dataset")
    parser.add_argument("--model_path", type=str, default=model_path, help="Path to fine-tuned model")
    parser.add_argument("--dataset_path", type=str, default=dataset_path, help="Path to ShareGPT JSON dataset")
    parser.add_argument("--image_base_path", type=str, default=dataset_path, help="Base path for images")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = QwenVLEvaluator(args.model_path, args.device)
    
    # Load dataset
    dataset = evaluator.load_sharegpt_dataset(args.dataset_path, args.image_base_path)
    
    # Limit samples if specified
    if args.max_samples:
        dataset = dataset[:args.max_samples]
    
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
    print(f"Response rate: {metrics['response_rate']:.2%}")
    print(f"Average prediction length: {evaluation_results['statistics']['average_prediction_length']:.1f} chars")
    print(f"Average ground truth length: {evaluation_results['statistics']['average_ground_truth_length']:.1f} chars")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main('/workspace/user_code/workspace/LLaMA-Factory/saved/qwen2_5vl-7b/lora/sft-0/checkpoint-940', '/workspace/user_code/workspace/llm-sft/data/pokemon')
