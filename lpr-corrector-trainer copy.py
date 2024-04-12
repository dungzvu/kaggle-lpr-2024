#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd

data_df = pd.read_csv("all_text_data.csv")
data_df = data_df.sample(frac=0.01).reset_index(drop=True)
# data_df = data_df.iloc[:1]


# aug more data

# In[15]:


get_ipython().system('pip install nlpaug')


# In[55]:


import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

augW_insert = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="insert",
    aug_min=1, aug_max=1, aug_p=0.1,
    device="cuda",
)

augW_sub = naw.ContextualWordEmbsAug(
    model_path='bert-base-cased',
    action="substitute",
    aug_min=1, aug_max=2, aug_p=0.3,
    device="cuda"
)

# augW_insert = naw.WordEmbsAug(
#     model_type='word2vec', model_path='/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin',
#     action="insert",
#     aug_min=1, aug_max=1, aug_p=0.1,
# )

# augW_sub = naw.WordEmbsAug(
#     model_type='word2vec', model_path='/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin',
#     action="substitute",
#     aug_min=1, aug_max=2, aug_p=0.3,
# )

augS_insert = nas.ContextualWordEmbsForSentenceAug(
    model_path='distilgpt2',
    max_length=100,
)


# In[49]:


import random
import re
random.seed(42)

text = "Rewrite this as inspired by Starwar"

def get_variants(text, n_max=10, step=2):
    finals = []
    queue = [ text ]
    while (len(finals) + len(queue) < n_max and len(queue) > 0) or text in queue:
        t = queue.pop(0)
        c_ = 0
        while c_ < step:
            gen = augW_insert.augment(augW_sub.augment(t)[0])[0]
#             if random.random() < 0.1:
#                 t = augS_insert.augment(t)[0]
            if gen not in queue and gen not in finals and gen != t:
                queue.append(gen)
                c_ += 1
        if t != text:
            finals.append(t)
    finals += queue

    finals = list(set(finals) - set([text]))
    finals = finals[:n_max]
    
    # clean
    try:
        finals = [re.sub(r"[\.+\"\'\:\?]", "", t) for t in finals]
    except:
        pass
    return finals

print(get_variants(text, n_max=1))


# In[56]:


import tqdm

ls = []
for i, row in tqdm.tqdm(data_df.iterrows(), total=len(data_df)):
    v = get_variants(row['text'], n_max=1)
    r_new = {
        f"text_{i}": v[i] for i in range(len(v))
    }
    ls.append(r_new)
    
df_new = pd.DataFrame(ls)
df_new["supported_text"] = data_df["supported_text"]
df_new["text"] = data_df["text"]

df_new.to_csv("data.csv", index=False)
df_new.head(2)

