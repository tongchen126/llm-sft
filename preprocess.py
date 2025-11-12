from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm
import random
from collections import defaultdict
import os

from dataset import get_label, preprocess_cot_prompt, get_tag, preprocess_record_hook
from utils import plot_label_distribution, gpt_api, image_to_base64

def get_role_message(messages,role):
       return [msg for msg in messages if msg["role"] == role]

def convert_to_cot(messages, image_paths, TAG, model="gpt-4o", dataset_name = None):
       """
       Convert a user → assistant pair into a chain-of-thought format.
       If images exist, they will be base64-encoded and sent to GPT-4o.
       """
       api_system_prompt = \
       """You are a helpful annotator. You are presented with the dialog between a user and an assistant.
       Rewrite the assistant's answer to include explicit reasoning steps. 
       You first give explicit reasoning steps,
       then followed by a label, which is the shortest answer, then followed by explanation. 
       You answer with the following format:
       <reasoning> [Step by step reasoning...] </reasoning>
       <label>[The short answer]</label> <explanation>[Explain the short answer]</explanation>."""

       result_system_prompt = """You are a helpful reasoning assistant. Always think step by step before answering. 
       You first give explicit reasoning steps,
       then followed by a label, which is the shortest answer, then followed by explanation. 
       You answer with the following format:
       <reasoning> [Step by step reasoning...] </reasoning>
       <label>[The short answer]</label> <explanation>[Explain the short answer]</explanation>."""

       if dataset_name is not None:
              api_system_prompt, result_system_prompt = preprocess_cot_prompt(dataset_name)

       user_msg = get_role_message(messages,"user")[0]["content"]
       assistant_msg = get_role_message(messages,"assistant")[0]["content"]

       user_content = []
       if user_msg:
              user_content.append({"type": "text", "text": f"User asked: {user_msg}"})
       if image_paths:
              for path in image_paths:
                     img_b64 = image_to_base64(path)
                     user_content.append({
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                            }
                     })
       user_content.append({
              "type": "text",
              "text": f"Original Assistant Answer: {assistant_msg}"
       })

       messages=[
              {"role": "system", "content": api_system_prompt},
              {"role": "user", "content": user_content}
       ]
       reply = gpt_api(model=model,messages=messages)

       new_messages = [
              {TAG.ROLE_TAG: TAG.SYSTEM_TAG,     TAG.CONTENT_TAG: result_system_prompt},
              {TAG.ROLE_TAG: TAG.USER_TAG,       TAG.CONTENT_TAG: user_msg},
              {TAG.ROLE_TAG: TAG.ASSISTANT_TAG,  TAG.CONTENT_TAG: reply}
       ]

       return new_messages

def conv_role(from_str, TAG):
       # dataset uses e.g. "human" and "gpt" in conversations -> map to sharegpt roles
       m = from_str.lower()
       if m in ("human","user","human:"):
              return TAG.USER_TAG
       if m in ("gpt","assistant","ai"):
              return TAG.ASSISTANT_TAG
       if m in ("system",):
              return TAG.SYSTEM_TAG
       return TAG.USER_TAG

def split_train_eval(data_path, eval_ratio, max_repeat_each_label=5):
       """
       Split dataset into train and eval sets with no label leakage.

       Args:
              data_path: Path to data.json
              eval_ratio: Float between 0 and 1, proportion of data for eval set

       Returns:
              tuple: (train_records, eval_records)
       """
       with open(data_path, 'r', encoding='utf-8') as f:
              records = json.load(f)

       if not 0 <= eval_ratio <= 1:
              raise ValueError("eval_ratio must be between 0 and 1")

       if not records:
              return {'data.json': [], 'data_eval.json': []}

       # Group records by label
       label_to_records = defaultdict(list)
       for record in records:
              label = record['label']
              label_to_records[label].append(record)

       label_to_del = []
       for key, val in label_to_records.items():
              if len(val) > max_repeat_each_label:
                     label_to_del.append(key)

       for key in label_to_del:
              label_to_records.pop(key, None)

       # Get all unique labels and shuffle them
       labels = list(label_to_records.keys())
       random.shuffle(labels)
       print(f"All labels:\n{labels}")

       # Calculate target number of eval records
       total_records = len(records)
       target_eval_count = int(total_records * eval_ratio)

       # Assign labels to eval set until we reach target count
       eval_labels = set()
       eval_count = 0

       train_eval_labels = set()

       for label in labels:
              label_count = len(label_to_records[label])
              if eval_count < target_eval_count:
                     eval_labels.add(label)
                     eval_count += label_count
              elif label_count >= 2:
                     train_eval_labels.add(label)

       # Split records based on label assignment
       train_records = []
       train_eval_records = []
       eval_records = []

       for label, label_records in label_to_records.items():
              if label in eval_labels:
                     eval_records.extend(label_records)
              elif label in train_eval_labels:
                     train_eval_records.extend(label_records[:1])
                     train_records.extend(label_records[1:])
              else:
                     train_records.extend(label_records)

       # Shuffle the final datasets
       random.shuffle(train_records)
       random.shuffle(train_eval_records)
       random.shuffle(eval_records)

       return {'data.json': train_records, 'data_train_eval.json': train_eval_records, 'data_eval.json': eval_records}

def extract_label(dataset_name, record, TAG):
       """
       Extract the label from assistant message content.
       Returns the text before the first colon in the assistant's message.

       Args:
              record: Dict containing 'messages' list with role and content
              
       Returns:
              str: The label (text before first ':') or None if not found
       """
       # Get messages list
       messages = record.get(TAG.MESSAGE_KEY, [])

       # Find assistant message
       for message in messages:
              if message.get(TAG.ROLE_TAG) == TAG.ASSISTANT_TAG:
                     content = message.get(TAG.CONTENT_TAG, '')
                     # Split by ':' and get first part
                     return get_label(dataset_name, content)

       return None

def conv_dataset(out_path = "data/pokemon",data_name = "llamafactory/pokemon-gpt4o-captions",message_name="conversations",to_cot=False, dataset_name = None, reasoning_model = 'gpt-4o'):
       TAG = get_tag('sharegpt')
       ds = load_dataset(data_name)["train"]  # or appropriate split

       out_dir = Path(out_path)
       image_default_dir = "images"
       img_out_dir = out_dir / image_default_dir
       img_out_dir.mkdir(parents=True, exist_ok=True)

       records = []
       for i, ex in tqdm(enumerate(ds)):
       # ex likely has 'conversations' (list of dict {from, value, lang}) and 'images' (list)
              conv = ex.get(message_name, [])
              messages = []
              for item in conv:
                     # some datasets use 'from'/'value'; adapt if keys differ
                     role = conv_role(item.get("from", item.get("role", TAG.USER_TAG)), TAG)
                     content = item.get("value", item.get("content", "") )
                     messages.append({TAG.ROLE_TAG: role, TAG.CONTENT_TAG: content})
              images = ex.get("images", [])  # HF image objects or URLs
       # convert HF image object to url or string if necessary
              img_urls = []
              for j, im in enumerate(images):
                     if isinstance(im, dict) and "path" in im:
                            img_urls.append(im["path"])
                     elif hasattr(im, "save"):
                            img_path = img_out_dir / f"{i}_{j}.png"
                            im.save(img_path)
                            img_urls.append(str(Path(image_default_dir) / f"{i}_{j}.png"))

              cur_record = {"id": i, TAG.MESSAGE_KEY: messages, TAG.IMAGE_KEY: img_urls}

              if dataset_name is not None:
                     label = extract_label(dataset_name, cur_record, TAG)
                     if label is not None:
                            if isinstance(label, list):
                                   label = label[0] # Take the first label, the original ground truth.
                            cur_record['label'] = label
                     else:
                            continue

              if to_cot:
                     try:
                            cur_record[TAG.MESSAGE_KEY] = convert_to_cot(cur_record[TAG.MESSAGE_KEY], [str(out_dir / i) for i in img_urls], TAG, model = reasoning_model, dataset_name = dataset_name)
                     except Exception as e:
                            print(e)
                            continue
              cur_record = preprocess_record_hook(dataset_name, TAG, cur_record, to_cot)
              records.append(cur_record)

       with open(out_dir / "data_all.json", "w", encoding="utf-8") as f:
              json.dump(records, f, ensure_ascii=False, indent=2)
              print("wrote ", out_dir / "data_all.json")

if __name__ == '__main__':
       out_path = "data/pokemon1_cot2/"
       conv_dataset(out_path=out_path, to_cot=True, dataset_name = 'pokemon',reasoning_model='gpt-5-chat')

       # write json list
       data = split_train_eval(os.path.join(out_path, 'data_all.json'), 0.1)
       for key, val in data.items():
              with open(os.path.join(out_path, key), "w", encoding="utf-8") as f:
                     json.dump(val, f, ensure_ascii=False, indent=2)
              plot_label_distribution(os.path.join(out_path, key), save_path = os.path.join(out_path, key.replace('.json', '.png')))
