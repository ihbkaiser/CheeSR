import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class CodeT5Embedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Salesforce/codet5p-770m", 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "Salesforce/codet5p-770m",
            trust_remote_code=True
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def embed(self, code: str) -> np.ndarray:
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked_embeddings = embeddings * attention_mask
        pooled = masked_embeddings.sum(1) / attention_mask.sum(1)
        
        return pooled.cpu().numpy().flatten()
    
class CodeT5Embedder110M:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Salesforce/codet5p-110m-embedding", 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "Salesforce/codet5p-110m-embedding",
            trust_remote_code=True
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def embed(self, code: str) -> np.ndarray:
        tok = self.tokenizer.encode(code, return_tensors="pt").to(self.device)
        embedding = self.model(tok)[0]
        return embedding.mean(dim=1).cpu().numpy().flatten()