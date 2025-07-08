import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    """Abstract base class for all embedding models."""
    @abstractmethod
    def encode(self, texts, **kwargs):
        """Encodes a list of texts into embeddings."""
        pass

class SentenceTransformerModel(EmbeddingModel):
    """Wrapper for open-source sentence-transformer models."""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers: `pip install sentence-transformers`")
        self.model = SentenceTransformer(model_name)
        print(f"Initialized embedding model: {model_name}")

    def encode(self, texts, **kwargs):
        return self.model.encode(texts, **kwargs)

class OpenAIEmbeddingModel(EmbeddingModel):
    """
    Wrapper for OpenAI's embedding models - requires an API key
    Processes requests in batches to avoid token limits
    """
    def __init__(self, model_name='text-embedding-ada-002', api_key=None):
        try:
            import openai
        except ImportError:
            raise ImportError("Need to install OpenAI library: `pip install openai`")
        
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OpenAI API key not found - Please set the OPENAI_API_KEY environment variable or pass it as an argument")
            
        self.model_name = model_name
        print(f"Initialized embedding model: {model_name}")

    def encode(self, texts, **kwargs):
        """
        Encodes texts using the OpenAI API, processing in batches 
        NOTE: NOT free
        """
        batch_size = 200
        all_embeddings = []
        
        print(f"Sending {len(texts)} texts to OpenAI API in batches of {batch_size}...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Getting OpenAI Embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [text.replace("\n", " ") for text in batch_texts]
            
            response = self.client.embeddings.create(input=batch_texts, model=self.model_name)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
        return np.array(all_embeddings)

EMBEDDING_CACHE_DIR = './embedding_cache'
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

class DynamicSampler:
    """
    Handles logic for finding semantically similar few-shot examples
    Generates and caches embeddings for datasets to avoid re-computation
    """
    def __init__(self, few_shot_pool_path, data_file_path, embedding_model_name='all-MiniLM-L6-v2', api_key=None):
        self.few_shot_pool_path = few_shot_pool_path
        self.data_file_path = data_file_path
        
        if "text-embedding" in embedding_model_name:
            self.embedding_model = OpenAIEmbeddingModel(
                model_name=embedding_model_name, 
                api_key=api_key
            )
        else:
            self.embedding_model = SentenceTransformerModel(model_name=embedding_model_name)
        
        print("Loading and embedding training data...")
        self.few_shot_data, self.few_shot_embeddings = self._load_and_embed(self.few_shot_pool_path)
        
        print("Loading and embedding test data...")
        self.dataset, self.data_embeddings = self._load_and_embed(self.data_file_path)

        print("Building FAISS index...")
        self.index = self._build_faiss_index(self.few_shot_embeddings)

    def _get_cache_path(self, file_path):
        """
        Generates path to cache embeddings based on model name
        """
        model_name_slug = self.embedding_model.__class__.__name__.lower()
        basename = os.path.basename(file_path).replace('.json', f'_{model_name_slug}.npy')
        return os.path.join(EMBEDDING_CACHE_DIR, basename)

    def _load_and_embed(self, file_path):
        """
        Loads dataset and generates embeddings, uses cache if available
        """
        cache_path = self._get_cache_path(file_path)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if os.path.exists(cache_path):
            print(f"Loading embeddings from cache: {cache_path}")
            embeddings = np.load(cache_path)
        else:
            print(f"Generating embeddings for {file_path}...")
            questions = [item['question'] for item in data]
            embeddings = self.embedding_model.encode(questions, show_progress_bar=True, convert_to_numpy=True)
            print(f"Saving embeddings to cache: {cache_path}")
            np.save(cache_path, embeddings)
            
        return data, embeddings

    def _build_faiss_index(self, embeddings):
        """
        Builds FAISS index for fast k-NN search
        """
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index

    def get_dynamic_few_shot_examples(self, k):
        """
        Find top k similar questions from the few-shot pool for each question
        """
        print(f"Finding top {k} similar examples for each test question...")
        data_embeddings_f32 = self.data_embeddings.astype('float32')
        distances, indices = self.index.search(data_embeddings_f32, k + 1)
        
        few_shot_map = {}
        for i in tqdm(range(len(self.dataset)), desc="Mapping few-shot examples"):
            data_item_id = self.dataset[i]['id']
            neighbor_indices = indices[i]
            
            few_shot_examples = []
            for idx in neighbor_indices:
                if len(few_shot_examples) < k:
                    if self.few_shot_data[idx]['id'] != data_item_id:
                        few_shot_examples.append(self.few_shot_data[idx])
            
            few_shot_map[data_item_id] = few_shot_examples
            
        return few_shot_map
