import json
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm
import numpy as np
from typing import List, Dict
from pathlib import Path
from torch.utils.data import Dataset
from peft import PeftModel

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

    def transform_conversation_sharegpt(self, input_data, TAG, base_url, system_message=None, skip_role = []):
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

    def get_gt_sharegpt(self, input_data, GT_ROLE, TAG):
        for message in input_data[TAG.MESSAGE_TAG]:
            role = message[TAG.ROLE_TAG]
            content = message[TAG.CONTENT_TAG]
            if role == GT_ROLE:
                return content

    def load_dataset(self, dataset_path: str, image_base_path: str = None, json_name = 'data.json', dataset_type = 'sharegpt') -> List[Dict]:
        assert(dataset_type == 'sharegpt')
        TAG = TAGS('images', 'messages', "assistant", "user", "system", "role", "content", "<image>")
        system_message = "You are a helpful assistant. You answer user's question with a standard format: [Short Answer]: [Explanation]"
        with open(str(Path(dataset_path, json_name)), 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Process dataset to handle image paths
        processed_dataset = {'processed':[],'original': [], 'gt': []}
        for item in dataset:
            gt = self.get_gt_sharegpt(item, TAG.ASSISTANT_TAG, TAG)
            original_item = self.transform_conversation_sharegpt(item, TAG, image_base_path, system_message = None)

            processed_item = self.transform_conversation_sharegpt(item, TAG, image_base_path, system_message = system_message, skip_role = [TAG.ASSISTANT_TAG])
            processed_chat = self.processor.apply_chat_template(processed_item, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")

            processed_dataset['original'].append(original_item)
            processed_dataset['processed'].append(processed_chat)
            processed_dataset['gt'].append(gt)
            
        processed_dataset = LLMDataset(processed_dataset, ["processed","original","gt"])

        return processed_dataset
    
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
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        return 0
