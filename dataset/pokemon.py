class PokemonHelper:
    @staticmethod
    def get_label(content):
        if ':' in content:
            label = content.split(':', 1)[0].strip()
            return label
        elif "：" in content:
            label = content.split('：', 1)[0].strip()
            return label
        return ''
    @staticmethod
    def construct_prompt(dataset = None):
        system_message = "You are a helpful assistant. You answer user's question with a standard format,\
                which consists of a short answer, and an explanation, with a colon separate them (<answer>: <explanation>)." + "A sample answer is: \
                Yamask: A ghostly, shadowy entity with a black body and red, slitted eyes that evoke an eerie aura."

        if dataset is not None:
            label_string = " All the possible answers include: " + ', '.join([i['label'] for i in dataset])
            system_message = system_message + label_string

        return system_message