# save_as_sharegpt.py
from datasets import load_dataset
import json
from pathlib import Path
from openai import OpenAI
import base64
from tqdm import tqdm

def image_to_base64(image_path):
       with open(image_path, "rb") as f:
              image_base64 = base64.b64encode(f.read()).decode("utf-8")
       return image_base64

def gpt_api(model,system=None,user=None,image_path=None,messages=None):
       token = "irk4CnzkwB6dCF8VOOBxI2V3@2700"
       url = "http://v2.open.venus.oa.com/llmproxy"

       # 构建请求数据
       if messages is None:
              with open(image_path, "rb") as f:
                     image_base64 = base64.b64encode(f.read()).decode("utf-8")

              messages =  [
                     {
                            "role": "system",
                            "content": system
                     },
                     {
                            "role": "user",
                            "content": [
                            {
                                   "type": "text", 
                                   "text": user
                            },
                            {
                                   "type": "image_url",
                                   "image_url": {
                                   "url": f"data:image/png;base64,{image_base64}"
                                   }
                            }
                            ] 
                     }
              ]

       client = OpenAI(
              base_url=url,
              api_key=token
       )

       response = client.chat.completions.create(
              model=model,
              messages=messages,
       )

       reply = response.choices[0].message.content
       return reply

def get_role_message(messages,role):
       return [msg for msg in messages if msg["role"] == role]

def convert_to_cot(messages, image_paths, model="gpt-4o"):
       """
       Convert a user → assistant pair into a chain-of-thought format.
       If images exist, they will be base64-encoded and sent to GPT-4o.
       """
       system_prompt = \
       """You are a helpful annotator. You are presented with the dialog between a user and an assistant.
       Rewrite the assistant's answer to include explicit reasoning steps. Use the following format: 
       <reasoning> [Step by step reasoning...] </reasoning> <final>[Concise final answer]</final>
       """
       result_system_prompt = """You are a helpful reasoning assistant. Always think step by step before answering. Format your response as: <reasoning> [step by step reasoning...] </reasoning> <final> [concise final answer only] </final> """
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
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_content}
       ]
       reply = gpt_api(model=model,messages=messages)

       new_messages = [
              {"role":"system","content": result_system_prompt},
              {"role": "user", "content": user_msg},
              {"role": "assistant", "content": reply}
       ]

       return new_messages

def conv_role(from_str):
    # dataset uses e.g. "human" and "gpt" in conversations -> map to sharegpt roles
       m = from_str.lower()
       if m in ("human","user","human:"):
              return "user"
       if m in ("gpt","assistant","ai"):
              return "assistant"
       if m in ("system",):
              return "system"
       return "user"

def conv_dataset(out_path = "data/pokemon",data_name = "llamafactory/pokemon-gpt4o-captions",message_name="conversations",to_cot=False):
       ds = load_dataset(data_name)["train"]  # or appropriate split

       out_dir = Path(out_path)
       image_default_dir = "images"
       out_file = out_dir / "data.json"
       img_out_dir = out_dir / image_default_dir
       img_out_dir.mkdir(parents=True, exist_ok=True)

       records = []
       for i, ex in tqdm(enumerate(ds)):
       # ex likely has 'conversations' (list of dict {from, value, lang}) and 'images' (list)
              conv = ex.get(message_name, [])
              messages = []
              for item in conv:
                     # some datasets use 'from'/'value'; adapt if keys differ
                     role = conv_role(item.get("from", item.get("role", "user")))
                     content = item.get("value", item.get("content", "") )
                     messages.append({"role": role, "content": content})
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
              if to_cot:
                     messages = convert_to_cot(messages, [str(out_dir / i) for i in img_urls])
              records.append({"id": i, "messages": messages, "images": img_urls})
       # write json list
       with open(out_file, "w", encoding="utf-8") as f:
              json.dump(records, f, ensure_ascii=False, indent=2)

       print("wrote", out_file)
       
conv_dataset(out_path="data/pokemon_cot", to_cot=True)
