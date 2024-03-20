from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from models.utils import generate_gemma_prompt

class GemmaModel:
    def __init__(self, model_name, device="cuda"):
        self.device = device

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            quantization_config=quantization_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict_prompt(self, original_text, rewritten_text, max_length=100):
        prompt = generate_gemma_prompt(original_text, rewritten_text)
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            prompt_ids, 
            max_length=max_length, 
            do_sample=True, 
            temperature=0.7, 
            top_k=50, 
            top_p=0.92, 
            num_return_sequences=1
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
