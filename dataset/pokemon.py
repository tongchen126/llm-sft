def pokemon_get_label(content):
    if ':' in content:
        label = content.split(':', 1)[0].strip()
        return label
    elif "：" in content:
        label = content.split('：', 1)[0].strip()
        return label
    return ''

def pokemon_construct_prompt(dataset):
    label_string = ', '.join([i['label'] for i in dataset])
    system_message = "You are a helpful assistant. You answer user's question with a standard format,\
                which consists of a short answer, and an explanation, with a colon separate them (<answer>: <explanation>). A sample answer looks like this: \
                Egg: The image draws an egg as it has a round shape with light-yellow color. All the possible answers include: " + label_string
    return system_message
