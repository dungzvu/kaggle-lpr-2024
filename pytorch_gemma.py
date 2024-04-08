#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2024 Google LLC.

# In[ ]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://ai.google.dev/gemma/docs/pytorch_gemma"><img src="https://ai.google.dev/static/site-assets/images/docs/notebook-site-button.png" height="32" width="32" />View on ai.google.dev</a>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemma/docs/pytorch_gemma.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/google/generative-ai-docs/blob/main/site/en/gemma/docs/pytorch_gemma.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>

# # Gemma in PyTorch
# 
# This is a quick demo of running Gemma inference in PyTorch.
# For more details, please check out the Github repo of the official PyTorch implementation [here](https://github.com/google/gemma_pytorch).
# 
# **Note that**:
#  * The free Colab CPU Python runtime and T4 GPU Python runtime are sufficient for running the Gemma 2B models and 7B int8 quantized models.
#  * For advanced use cases for other GPUs or TPU, please refer to [README.md](https://github.com/google/gemma_pytorch/blob/main/README.md) in the official repo.

# ## Kaggle access
# 
# To login to Kaggle, you can either store your `kaggle.json` credentials file at
# `~/.kaggle/kaggle.json` or run the following in a Colab environment. See the
# [`kagglehub` package documentation](https://github.com/Kaggle/kagglehub#authenticate)
# for more details.

# In[4]:


import kagglehub

kagglehub.login()


# ## Install dependencies

# In[5]:


get_ipython().system('pip install -q -U immutabledict sentencepiece')


# ## Download model weights

# In[6]:


# Choose variant and machine type
VARIANT = '2b-it' #@param ['2b', '2b-it', '7b', '7b-it', '7b-quant', '7b-it-quant']
MACHINE_TYPE = 'cuda' #@param ['cuda', 'cpu']


# In[7]:


import os

# Load model weights
weights_dir = kagglehub.model_download(f'google/gemma/pyTorch/{VARIANT}')

# Ensure that the tokenizer is present
tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

# Ensure that the checkpoint is present
ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT}.ckpt')
assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'


# In[8]:


weights_dir


# ## Download the model implementation

# In[9]:


# NOTE: The "installation" is just cloning the repo.
get_ipython().system('git clone https://github.com/google/gemma_pytorch.git')


# In[10]:


import sys

sys.path.append('gemma_pytorch')


# In[11]:


from gemma_pytorch.gemma.config import get_config_for_7b, get_config_for_2b
from gemma_pytorch.gemma.model import GemmaForCausalLM


# ## Setup the model

# In[12]:


import torch

# Set up model config.
model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
model_config.tokenizer = tokenizer_path
model_config.quant = 'quant' in VARIANT

# Instantiate the model and load the weights.
torch.set_default_dtype(model_config.get_dtype())
device = torch.device(MACHINE_TYPE)
model = GemmaForCausalLM(model_config)
model.load_weights(ckpt_path)
model = model.to(device).eval()


# ## Run inference
# 
# Below are examples for generating in chat mode and generating with multiple
# requests.
# 
# The instruction-tuned Gemma models were trained with a specific formatter that
# annotates instruction tuning examples with extra information, both during
# training and inference. The annotations (1) indicate roles in a conversation,
# and (2) delineate turns in a conversation. Below we show a sample code snippet
# for formatting the model prompt using the user and model chat templates in a
# multi-turn conversation. The relevant tokens are:
# 
# - `user`: user turn
# - `model`: model turn
# - `<start_of_turn>`: beginning of dialogue turn
# - `<end_of_turn>`: end of dialogue turn
# 
# Read about the Gemma formatting for instruction tuning and system instructions
# [here](https://ai.google.dev/gemma/docs/formatting).

# In[13]:


# Generate with one request in chat mode

# Chat templates
USER_CHAT_TEMPLATE = '<start_of_turn>user\n{prompt}<end_of_turn>\n'
MODEL_CHAT_TEMPLATE = '<start_of_turn>model\n{prompt}<end_of_turn>\n'

# # Sample formatted prompt
# prompt = (
#     USER_CHAT_TEMPLATE.format(
#         prompt='What is a good place for travel in the US?'
#     )
#     + MODEL_CHAT_TEMPLATE.format(prompt='California.')
#     + USER_CHAT_TEMPLATE.format(prompt='What can I do in California?')
#     + '<start_of_turn>model\n'
# )
# print('Chat prompt:\n', prompt)

# model.generate(
#     [prompt, ]*10,
#     device=device,
#     output_len=100,
#     temperature=0.8,
# )


# In[37]:


from pathlib import Path

DATA_PATH = Path("./data")


# In[14]:


# Generate sample
# model.generate(
#     'Write a poem about an llm writing a poem.',
#     device=device,
#     output_len=60,
# )


# In[19]:


#import pandas as pd
#import tqdm
#
#df = pd.read_csv(DATA_PATH / "200px300t_data_predict.csv")
#df.head()
#
#rewrite_prompt = df['rewrite_prompt'].iloc[::30].tolist()
#rewrite_prompt
#
#prompt = (
#    USER_CHAT_TEMPLATE.format(
#        prompt='Rephase this sentence, startswith "rewrite, transform, convert ...": "{}"'
#    )
#    + '<start_of_turn>model\n'
#)
#
#rs = []
#for i in tqdm.tqdm(range(len(rewrite_prompt))):
#    p = prompt.format(rewrite_prompt[i])
#    results = model.generate(
#        [p, ] * 30,
#        device=device,
#        output_len=100,
#    )
#
#    for r in results:
#        try:
#            s = r.split(':\n\n')[1]
#            if s.startswith('**'):
#                s = s[2:]
#            rs.append(s)
#        except:
#            rs.append(r)
#
#
#df['rewrite_prompt'] = rs
#df.to_csv(DATA_PATH / '200px30t_data_predict_aug.csv', index=False)

import pandas as pd
import tqdm
df = pd.read_csv(DATA_PATH / 'validate_dataset.csv')


# In[38]:


import numpy as np

df.shape

df_splits = np.array_split(df, 10)


# In[39]:


prompt = (
    USER_CHAT_TEMPLATE.format(
        prompt='{}: """{}"""'
    )
    + '<start_of_turn>model\n'
)

def truncate_words(text, max_words):
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])

# batch = 4 ~ 25% faster on rtx 4090
batch_size = 4

for split_idx, df_s in enumerate(df_splits):
    print(f'Chunk {split_idx} ..')
    rows = []
    for idx in tqdm.tqdm(range(0, len(df_s), batch_size)):
        batch_df = df_s.iloc[idx:idx+batch_size]
        prompts = []
        for _, row in batch_df.iterrows():
            p = prompt.format(row['rewrite_prompt'], truncate_words(row['original_text'], 200))
            prompts += [p, ]

        try:
            results = model.generate(
                prompts,
                device=device,
                output_len=250,
            )
            i = 0
            for _, row in batch_df.iterrows():
                rs = results[i]
                i += 1
                row = {
                    'rewrite_prompt': row['rewrite_prompt'],
                    'original_text': row['original_text'],
                }
                row.update({
                    f'rewritten_text': rs,
                })
                rows.append(row)
        except Exception as e:
            print(f'Error: {e}')
            continue

    print(f'Saving chunk {split_idx} ..')
    df_s = pd.DataFrame(rows)
    df_s.to_csv(DATA_PATH / f'validate_dataset_chunk_{split_idx}.csv', index=False)


# In[ ]:


"Title: Impact of Tonic on Dorothy's Energy Levels: A Trend Analysis\n\nIntroduction:\nThis report analyzes the effects of daily tonic consumption on Dorothy's energy levels over two weeks.\n\nMethodology:\nDorothy consumed one bottle of XYZ brand tonic every morning for two weeks. Energy levels were rated on a scale of 1 to 10.\n\nResults:\nConsuming tonic significantly improved Dorothy's energy levels. The average rating increased from 4.5 to 7.8, indicating a notable upward trend in energy throughout the study.\n\nConclusion:\nRegular tonic consumption positively impacts Dorothy's energy levels. Further research is needed to understand the specific ingredients and mechanisms behind this effect.".split(" ").__len__()


# ## Learn more
# 
# Now that you have learned how to use Gemma in Pytorch, you can explore the many
# other things that Gemma can do in [ai.google.dev/gemma](https://ai.google.dev/gemma).
# See also these other related resources:
# 
# - [Gemma model card](https://ai.google.dev/gemma/docs/model_card)
# - [Gemma C++ Tutorial](https://ai.google.dev/gemma/docs/gemma_cpp)
# - [Gemma formatting and system instructions](https://ai.google.dev/gemma/docs/formatting)
