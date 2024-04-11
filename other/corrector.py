from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

gemma_model_name = ""

gemma_model = AutoModelForCausalLM.from_pretrained(
    gemma_model_name, 
    device_map="auto",
    quantization_config=quantization_config
)
gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)