def pokemon_get_label(content:)
    if ':' in content:
        label = content.split(':', 1)[0].strip()
        return label
    elif "：" in content:
        label = content.split('：', 1)[0].strip()
        return label
    return None