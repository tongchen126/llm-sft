import re

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
        if label == '':
            return None
        return label

    @staticmethod
    def construct_prompt(dataset = None):
        sample_answer = " A sample answer is: \
                Yamask: A ghostly, shadowy entity with a black body and red, slitted eyes that evoke an eerie aura."
        system_message = "You are a helpful assistant. You answer user's question with a standard format,\
                which consists of a short answer, and an explanation, with a colon separate them (<answer>: <explanation>)." + sample_answer

        if dataset is not None:
            label_string = " All the possible answers include: " + ', '.join([i['label'] for i in dataset])
            system_message = system_message + label_string

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