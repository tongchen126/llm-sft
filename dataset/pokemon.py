import re
import json
import os
from typing import Optional, Dict

class PokemonTranslator:
    """宝可梦名字中英文互译类"""

    def __init__(self, json_file: str = './dataset/pokemon_data.json'):
        """
        初始化翻译器
        :param json_file: JSON数据文件路径
        """
        self.json_file = json_file
        self.pokemon_dict = {}  # 英文 -> 中文
        self.reverse_dict = {}  # 中文 -> 英文
        self.generation_dict = {}  # 世代分类
        self.load_data()

    def load_data(self):
        """从JSON文件加载宝可梦数据"""
        try:
            if not os.path.exists(self.json_file):
                print(f"❌ 错误：找不到文件 {self.json_file}")
                return

            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 构建字典
            for generation, pokemons in data.items():
                self.generation_dict[generation] = pokemons
                for eng_name, chi_name in pokemons.items():
                    self.pokemon_dict[eng_name] = chi_name
                    self.reverse_dict[chi_name] = eng_name

            print(f"✅ 成功加载 {len(self.pokemon_dict)} 只宝可梦数据")

        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误：{e}")
        except Exception as e:
            print(f"❌ 加载数据时出错：{e}")

    def translate(self, name: str) -> str:
        """
        翻译宝可梦名字（中英互译）
        :param name: 宝可梦名字
        :return: 翻译结果
        """
        # 英文 -> 中文
        if name in self.pokemon_dict:
            return self.pokemon_dict[name]
        # 中文 -> 英文
        elif name in self.reverse_dict:
            return self.reverse_dict[name]
        else:
            return f"❌ 未找到 '{name}' 的翻译"
    def fuzzy_search(self, keyword: str) -> list:
        """
        模糊搜索宝可梦
        :param keyword: 搜索关键词
        :return: 匹配的宝可梦列表
        """
        results = []
        keyword_lower = keyword.lower()

        # 搜索英文名
        for eng_name, chi_name in self.pokemon_dict.items():
            if keyword_lower in eng_name.lower() or eng_name.lower() in keyword_lower:
                results.append(chi_name)

        # 搜索中文名
        for chi_name, eng_name in self.reverse_dict.items():
            if keyword in chi_name or chi_name in keyword:
                results.append(eng_name)

        return results

_PokemonTranslator = PokemonTranslator()
class PokemonHelper:
    @staticmethod
    def _split_str(content, split_str, index = 0):
        if split_str in content:
            return content.split(split_str, 1)[index].strip()
        return content

    @staticmethod
    def get_label(content):
        label = PokemonHelper._split_str(content, ':')
        label = PokemonHelper._split_str(label, '：')
        label = PokemonHelper._split_str(label, '.')
        label = PokemonHelper._split_str(label, '。')
        label = PokemonHelper._split_str(label, '(')
        label = PokemonHelper._split_str(label, '（')
        label = PokemonHelper._split_str(label, '只', 1)
        label = PokemonHelper._split_str(label, '是', 1)
        label = PokemonHelper._split_str(label, '的', 1)
        label = PokemonHelper._split_str(label, 'is', 1)
        label = PokemonHelper._split_str(label, 'is', 1)

        return [label, *_PokemonTranslator.fuzzy_search(label)]

    @staticmethod
    def construct_prompt(dataset = None):
        sample_answer = " A sample answer is: \
                Yamask: A ghostly, shadowy entity with a black body and red, slitted eyes that evoke an eerie aura."
        system_message = "You are a helpful assistant that answers which pokemon is it in the image provided by the user. \
                You answer user's question in a standard format,\
                which consists of a short answer to which pokemon is it, and an explanation, with a colon separating them (<which pokemon>: <explanation>). \
                The user may raise question either in English or Chinese, \
                and you must answer the pokemon name in the same language as the user."

        if dataset is not None:
            label_string = " All the possible pokemons that may occur are: " + ', '.join([i['label'] for i in dataset])
            system_message = system_message + label_string

#        system_message = system_message + sample_answer

        return system_message
    
    @staticmethod
    def preprocess_cot_prompt():
        sample_answer = \
        """ A sample answer is:
        <thinking> First, identify the creature in the image as a Bulbasaur from the Pokémon franchise.
        Then, observe its size and body structure, noticing it is small and quadrupedal.
        Next, examine its color and physical features, noting its blue-green body and dark patches.
        Now, look at its eyes, recognizing the sharp, triangular shape with red irises. Observe the large plant bulb on its back, recognizing it as thick and green.
        Finally, notice additional details like pointed, stubby legs with claws and ear-like protrusions. </thinking>

        <label>Bulbasaur</label>

        <explanation>A small, quadruped creature with a blue-green body, sharp triangular eyes with red irises, and noticeable dark patches on its skin.
        It has a plant bulb on its back, which is thick and green, signifying its Grass/Poison typing.
        The bulb is prominent and resembles a small cabbage or plant bud. The creature has pointed, stubby legs with claws and an ear-like protrusion on each side of its head.
        This description matches Bulbasaur.</explanation>
        """
        api_system_prompt = \
        """You are a helpful annotator. You are presented with the dialog between a user and an assistant.
        Rewrite the assistant's answer to include explicit reasoning steps. 
        You first give explicit reasoning steps,
        then followed by a label, which is the shortest answer, then followed by explanation. 
        You answer with the following format:
        <thinking> [Step by step reasoning...] </thinking>
        <label>[The short answer]</label> <explanation>[Explain the short answer]</explanation>.""" + sample_answer

        result_system_prompt = """You are a helpful reasoning assistant. Always think step by step before answering. 
        You first give explicit reasoning steps,
        then followed by a label, which is the shortest answer, then followed by explanation. 
        You answer with the following format:
        <thinking> [Step by step reasoning...] </thinking>
        <label>[The short answer]</label> <explanation>[Explain the short answer]</explanation>.""" + sample_answer

        return api_system_prompt, result_system_prompt

    @staticmethod
    def get_cot_label(content):
        tag_to_find = 'label'
        pattern = f"<{tag_to_find}>(.*?)</{tag_to_find}>"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def construct_cot_prompt(dataset = None):
        sample_answer = \
        """ A sample answer is:
        <thinking> First, identify the creature in the image as a Bulbasaur from the Pokémon franchise.
        Then, observe its size and body structure, noticing it is small and quadrupedal.
        Next, examine its color and physical features, noting its blue-green body and dark patches.
        Now, look at its eyes, recognizing the sharp, triangular shape with red irises. Observe the large plant bulb on its back, recognizing it as thick and green.
        Finally, notice additional details like pointed, stubby legs with claws and ear-like protrusions. </thinking>

        <label>Bulbasaur</label>

        <explanation>A small, quadruped creature with a blue-green body, sharp triangular eyes with red irises, and noticeable dark patches on its skin.
        It has a plant bulb on its back, which is thick and green, signifying its Grass/Poison typing.
        The bulb is prominent and resembles a small cabbage or plant bud. The creature has pointed, stubby legs with claws and an ear-like protrusion on each side of its head.
        This description matches Bulbasaur.</explanation>
        """

        system_message = """You are a helpful reasoning assistant. Always think step by step before answering. 
        You first give explicit reasoning steps,
        then followed by a label, which is the shortest answer, then followed by explanation. 
        You answer with the following format:
        <thinking> [Step by step reasoning...] </thinking>
        <label>[The short answer]</label> <explanation>[Explain the short answer]</explanation>.""" + sample_answer

        if dataset is not None:
            label_string = " All the labels include: " + ', '.join([i['label'] for i in dataset])
            system_message = system_message + label_string

        return system_message


class PokemonLabelHelper:
    @staticmethod
    def get_label(content):
        return PokemonHelper.get_label(content)

    @staticmethod
    def construct_prompt(dataset = None):
        system_message = "You are a helpful assistant that answers which pokemon is it in the image provided by the user. \
                You answer user's question of which pokemon is it in the image, and you just answer the name of the pokemon, \
                without any other explanation. \
                Please remember, the user may raise question either in English or Chinese, \
                and you must answer the pokemon name in the same language as the user's."

        if dataset is not None:
            label_string = " All the possible pokemons that may occur are: " + ', '.join([i['label'] for i in dataset])
            system_message = system_message + label_string

#        system_message = system_message + sample_answer

        return system_message

    def preprocess_record_hook(TAG, record, cot):
        assert(not cot)
        messages = record.get(TAG.MESSAGE_KEY, [])

        for message in messages:
            if message.get(TAG.ROLE_TAG) == TAG.ASSISTANT_TAG:
                content = message.get(TAG.CONTENT_TAG, '')
                message[TAG.CONTENT_TAG] = PokemonLabelHelper.get_label(content)

        return record
