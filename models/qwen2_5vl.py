from .base import BaseEvaluator

class QwenVLEvaluator(BaseEvaluator):
    def __init__(self, base_model, lora_path = None, device: str = "auto", **kwargs):
        super().__init__(base_model, lora_path, device, **kwargs)