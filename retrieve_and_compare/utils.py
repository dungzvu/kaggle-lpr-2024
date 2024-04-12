
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import transformers
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from pathlib import Path
from torch import nn
import torch


DATA_PATH = Path("./")


class Perplexity(nn.Module):
    def __init__(self, reduce: bool = True):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.reduce = reduce

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        perplexity = []
        for i in range(labels.shape[0]):
            perplexity.append(self.loss_fn(shift_logits[i], shift_labels[i]))
        perplexity = torch.stack(perplexity, dim=0)
        #perplexity = torch.exp(perplexity)
        if self.reduce:
            perplexity = torch.mean(perplexity)
        return perplexity 


class Config(transformers.configuration_utils.PretrainedConfig):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                json.dumps(value)
                setattr(self, key, value)
            except TypeError:
                # value was not JSON-serializable, skip
                continue
        super().__init__()


class Retriever:

    def __init__(self, cfg: Config, device=None):
        self.device = device

        self.cfg = cfg
        self.style_encoder = ""
        self.db_df = ""
        self.style_embeddings = np.array([])
        self.desc_bm25 = ""

        # Load the dictionary, tfidf_model, and bm25_index from file
        dictionary = Dictionary.load(DATA_PATH / "outputs/bm25_description/dict")
        tfidf_model = TfidfModel.load(DATA_PATH / "outputs/bm25_description/tfidf")
        bm25_index = SparseMatrixSimilarity.load(DATA_PATH / "outputs/bm25_description/bm25_index")

        self.bm25 = (dictionary, tfidf_model, bm25_index)

    def get_top_similary_with_embeddings(self, query_embedding, top_k=1):
        data_embedding = self.style_embeddings
        n_rows = data_embedding.shape[0]

        cosine_similarities = cosine_similarity(
            [query_embedding], 
            self.style_embeddings.reshape(-1, len(query_embedding))
        ).reshape(n_rows, -1) \
        .mean(axis=1)

        top_k_indices = np.argsort(cosine_similarities)[::-1][:top_k]
        return top_k_indices, cosine_similarities[top_k_indices]
    
    def get_top_bm25_ranking(self, query, top_k=1):
        dictionary, tfidf_model, bm25_index = self.bm25

        query = dictionary.doc2bow(query.split())
        query_tfidf = tfidf_model[query]
        bm25_scores = bm25_index[query_tfidf]

        top_k_indices = np.argsort(bm25_scores)[::-1][:top_k]
        return top_k_indices, bm25_scores[top_k_indices]

    def search(self, query: str, rewritten_text: str = None, rank_with_embedding_index=None, top_k=1):
        n_top = max(top_k, 200)

        target_embedding = self.style_encoder.encode(rewritten_text)

        top_rewritten_indices, top_rewritten_scores = self.get_top_similary_with_embeddings(target_embedding, n_top)
        top_desc_indices, top_desc_scores = self.get_top_bm25_ranking(query, n_top)

        top_indices = np.array()
        scores = np.array()

        return top_indices, scores
    
    def search_relative(self, search_index, top_k=1):
        indices = []
        return indices
    
    def search_prompts_by_value(self, value: str, top_k=1):
        indices = []
        return indices
    
    def get_rows_by_indices(self, indices):
        return self.db_df.iloc[indices]
    

class PromptRecoveryModel:
    def __init__(self, cfg: Config, device=None):
        self.device = device
        self.cfg = cfg

        self.perp = Perplexity()
        model = ""
        tokenizer = ""
        self.llm = (model, tokenizer)
        self.retriever = Retriever(cfg, device)

    def compute_perplexity(self, samples):
        perp, (model, tokenizer) = self.perp, self.llm

        perps = []
        with torch.no_grad():
            inputs = tokenizer(samples, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).to("cuda")
            output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            output = output.logits
            labels = inputs["input_ids"]
            labels.masked_fill_(~inputs["attention_mask"].bool(), -100)
            for j in range(len(samples)):
                p = perp(output[j].unsqueeze(0), labels[j].unsqueeze(0))
                perps.append(p.detach().cpu().float())
        return perps
    
    def compute_perplexity_from_indices(self, indices):
        rows = self.retriever.get_rows_by_indices(indices)

        # build list of prompt here
        samples = []

        scores = self.compute_perplexity(samples)
        return scores
    
    def llm__extract_info(self, original_text: str, rewritten_text: str):
        model, tokenizer = self.llm

        # TODO: Build chat here to extract valuable information
        chat = [

        ]

        model_inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)
        generated_ids = model.generate(
            model_inputs, 
            max_new_tokens=30, 
            do_sample=True,
            temperature=0.72,
            top_p=0.8,
            top_k=10,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(generated_ids[0][len(model_inputs[0]):])

        # TODO: parse decoded answer here

        queries = []
        value = ""
        return queries, value
    
    def predict(self, original_text: str, rewritten_text: str, top_k=[1, 2]):
        # Extract info from llm
        queries, value = self.llm__extract_info(original_text, rewritten_text)

         # query by value
        if value:
            value_indices = self.retriever.search_prompts_by_value(value)
        else:
            value_indices = []

        indices = [] + value_indices

        # query the most relevant predefined prompts
        top_indices, retrieve_scores = self.retriever.search(
            queries, 
            rewritten_text, 
            top_k=top_k[0],
        )
        last_n_indices = len(indices)
        indices = np.unique(np.concatenate([indices, top_indices]))[:last_n_indices + top_k]
        new_indices = indices[last_n_indices:]
        new_scores = retrieve_scores[np.where(np.isin(top_indices, new_indices))]

        # run secondary search
        secondary_indices = []
        secondary_scores = []
        for query_index, query_score in zip(new_indices, new_scores):
            top_indices, scores = self.retriever.search_relative(
                top_k=top_k,
                rank_with_embedding_index=query_index,
            )
            scores = scores * query_score
            secondary_indices.extend(top_indices)
            secondary_scores.extend(scores)
        
        # find top_k in secondary search
        sorted_scores_idx = np.argsort(secondary_scores)[::-1]
        secondary_indices = np.unique(np.array(secondary_indices)[sorted_scores_idx])

        indices = np.unique(np.concatenate([indices, secondary_indices]))[:top_k[0] + top_k[1]]

        # after find all relevants, check prompt perplexity
        perplexities = self.compute_perplexity_from_indices(indices)
        top_indices = np.argsort(perplexities)[::-1]

        best_idx = top_indices[0]

        # return prompt here
        rewritten_prompt = ""
        return rewritten_prompt

       
