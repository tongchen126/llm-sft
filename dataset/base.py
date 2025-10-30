import json
from tqdm import tqdm
from typing import List, Dict
from pathlib import Path
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

def load_dataset(self, dataset_path: str, image_base_path: str = None, json_name = 'data.json', dataset_type = 'sharegpt', max_samples = None) -> List[Dict]:
    if (dataset_type == 'sharegpt'):
        TAG = TAGS('images', 'messages', "assistant", "user", "system", "role", "content", "<image>")
        system_message = "You are a helpful assistant. You answer user's question with a standard format: [Short Answer]: [Explanation]"
        with open(str(Path(dataset_path, json_name)), 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Process dataset to handle image paths
        if max_samples is not None:
            dataset = dataset[:max_samples]
        processed_dataset = {'processed':[],'original': [], 'gt': []}
        for item in tqdm(dataset):
            gt = self.get_gt_sharegpt(item, TAG.ASSISTANT_TAG, TAG)
            original_item = self.transform_conversation_sharegpt(item, TAG, image_base_path, system_message = None)

            processed_item = self.transform_conversation_sharegpt(item, TAG, image_base_path, system_message = system_message, skip_role = [TAG.ASSISTANT_TAG])
            processed_chat = self.processor.apply_chat_template(processed_item, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")

            processed_dataset['original'].append(original_item)
            processed_dataset['processed'].append(processed_chat)
            processed_dataset['gt'].append(gt)
            
        processed_dataset = LLMDataset(processed_dataset, ["processed","original","gt"])

        return processed_dataset
    else:
        processed_dataset = {'processed':[],'original': [], 'gt': []}
        processed_dataset = LLMDataset(processed_dataset, ["processed","original","gt"])
        return processed_dataset