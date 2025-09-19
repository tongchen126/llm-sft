# save_as_sharegpt.py
from datasets import load_dataset
import json
from pathlib import Path

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

def conv_dataset(out_path = "data/pokemon",data_name = "llamafactory/pokemon-gpt4o-captions",message_name="conversations"):
       ds = load_dataset(data_name)["train"]  # or appropriate split

       out_dir = Path(out_path)
       out_file = out_dir / "data.json"
       img_out_dir = out_dir / "images"
       img_out_dir.mkdir(parents=True, exist_ok=True)

       records = []
       for i, ex in enumerate(ds):
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
                            img_urls.append(str(Path("images") / f"{i}_{j}.png"))
              records.append({"id": i, "messages": messages, "images": img_urls})
       # write json list
       with open(out_file, "w", encoding="utf-8") as f:
              json.dump(records, f, ensure_ascii=False, indent=2)

       print("wrote", out_file)
       
conv_dataset()