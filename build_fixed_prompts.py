#!/usr/bin/env python
# # Ask gemma to build dataset

# In[1]:


get_ipython().system('pip install bitsandbytes accelerate transformers')

import os

os.environ["HF_TOKEN"] = "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"


# In[2]:

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", quantization_config=quantization_config)


# In[32]:


tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

def ask_gemma(input_text, max_new_tokens=100):
    input_ids = tokenizer(input_text, padding=True, return_tensors="pt")

    outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    results = [
        tokenizer.decode(outputs[i][len(input_ids[i]):], skip_special_tokens=True)
        for i in range(len(input_text))
    ]
    if isinstance(input_text, list):
        return results
    return results[0]


# In[38]:


import tqdm
import numpy as np
import pandas as pd

# load data first
df_core = pd.read_csv("core_objectives.csv", keep_default_na=False)

jsondata = [
  "Georgette, a loving mother, always puts her family's needs first. From waking up early to prepare breakfast to tucking her children into bed at night, she is the heart of their home. Her warm hugs and encouraging words make every day brighter. Georgette's unwavering love and dedication make her an incredible mother.",
  "Hey Mari! Just wanted to share something funny that happened today. So, I was telling my friends about that crazy party we went to last weekend, and you know how I tend to exaggerate things? Well, I may have added a little embellishment to the story. They were cracking up, but I couldn't help but laugh at myself too. Anyway, hope you're having a great day! Let's catch up soon.",
  "Jerry, a talented musician, nervously stepped onto the stage. As he began to play, his fingers stumbled, and the melody turned into a jumbled mess. The audience fell silent, disappointment filling the air. But Jerry didn't let this failure define him; he practiced harder and returned to the stage stronger than ever.",
  "Ladies and gentlemen, thank you for joining us today. I stand before you to shed light on the fascinating puffin. Found in the North Atlantic, these adorable birds are known for their colorful beaks and exceptional diving skills. Let's celebrate the puffin's resilience and conservation efforts. Together, we can ensure a bright future for these magnificent creatures. Thank you.",
  "Dear Mr. Johnson,\\n\\nI hope this email finds you well. Attached is the invoice for the recent services provided by Gale's Plumbing. The total amount due is $250, which includes the cost of labor and materials. Please review the invoice and kindly make the payment within 14 days. If you have any questions or concerns, feel free to reach out. Thank you for your business!\\n\\nBest regards,\\nEmily Smith\\nGale's Plumbing",
  "Ladies and gentlemen, thank you for being here today. I want to address a common issue we all face: chafing. Whether it's during exercise or everyday activities, chafing can be uncomfortable and irritating. But fear not! With the help of our new product, \"ChafeAway,\" you can bid farewell to chafing forever. Say goodbye to discomfort and hello to smooth, irritation-free skin. Try \"ChafeAway\" today and experience the difference for yourself."
]

ls = []
for i, row in tqdm.tqdm(df_core.iterrows(), total=len(df_core)):
    explain_prompt = row["explain_prompt"]
    # if explain_prompt == float('nan'):
    #     explain_prompt = None
        
    # print(f"explain {explain_prompt}, {type(explain_prompt)}")
    ls_prompt = [] if not explain_prompt else [ explain_prompt ]
    ls_prompt += [
        '{}: """{}"""'.format(row["rewrite_prompt"], text) for text in jsondata
    ]
    # print(ls_prompt)
    batch_outputs = ask_gemma(ls_prompt, max_new_tokens=200)
    desc, ls_rewritten = ("", batch_outputs) if not explain_prompt else (batch_outputs[0], batch_outputs[1:])
    
    d = {
        "type": row["type"],
        "objective": row["objective"],
        "description": desc,
    }
    d.update({
        f"rewritten_{i}": ls_rewritten[i] for i in range(len(ls_rewritten))
    })
    ls.append(d)

df_final = pd.DataFrame(ls)
df_final.to_csv("final_objectives.csv", index=False)


# In[39]:


df_final.iloc[0]['rewritten_1']

