import re
from typing import List

from swift.plugin import ORM, orms
from rapidfuzz.distance import Levenshtein

def _cal_exact(prediction_label, reference_label):
    diff = Levenshtein.distance(prediction_label, reference_label)
    diff = (len(reference_label) - diff) if (len(reference_label) - diff) > 0 else 0
    exact_metric = diff / len(reference_label)
    return exact_metric

def _get_cot_label(content):
    tag_to_find = 'label'
    pattern = f"<{tag_to_find}>(.*?)</{tag_to_find}>"
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else ""

class PokemonGRPOAccuracy(ORM):
    def __call__(self, completions, label, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            label (list[str]): Ground Truths.
        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for content, sol in zip(completions, label):
            answer = _get_cot_label(content)
            reward = _cal_exact(answer, sol)
            rewards.append(reward)
            #print(f"{answer}:{sol}:{reward}")
        return rewards

orms['pokemon_grpo_acc'] = PokemonGRPOAccuracy

class PokemonGRPOFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for content in completions:
            correct_count = 0
            
            # Check for <think> bracket
            if re.search(r'<thinking>.*?</thinking>', content, re.DOTALL):
                correct_count += 1
            
            # Check for <label> bracket
            if re.search(r'<label>.*?</label>', content, re.DOTALL):
                correct_count += 1
            
            # Check for <explanation> bracket
            if re.search(r'<explanation>.*?</explanation>', content, re.DOTALL):
                correct_count += 1
            
            # Bonus: check if all three are present in correct order
            full_pattern = r'<thinking>.*?</thinking>.*?<label>.*?</label>.*?<explanation>.*?</explanation>'
            if re.search(full_pattern, content, re.DOTALL):
                reward = 1.0  # Full reward for correct order
            else:
                reward = correct_count / 3.0  # Partial reward
            #print(f"-----------------------------------------\n\
            #    {content}:\n{reward}\n------------------------------------\n")
            rewards.append(reward)
        
        return rewards

orms['pokemon_grpo_format'] = PokemonGRPOFormat
