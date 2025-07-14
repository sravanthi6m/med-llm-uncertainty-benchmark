import os
import json
import faiss
import random
import numpy as np

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
    def __init__(self, data_file_path, few_shot_pool_path_1, embedding_model: EmbeddingModel, few_shot_pool_path_2=None):
        self.embedding_model = embedding_model
                
        print("Loading and embedding test data...")
        self.dataset, self.data_embeddings = self._load_and_embed(data_file_path)

        print("Loading and embedding few-shot pool 1...")
        self.pool1_data, self.pool1_embeddings = self._load_and_embed(few_shot_pool_path_1)
        self.pool1_index = self._build_faiss_index(self.pool1_embeddings)

        self.pool2_data = None
        self.pool2_embeddings = None
        self.pool2_index = None
        if few_shot_pool_path_2 and os.path.exists(few_shot_pool_path_2):
            print("Loading and embedding few-shot pool 2...")
            self.pool2_data, self.pool2_embeddings = self._load_and_embed(few_shot_pool_path_2)
            self.pool2_index = self._build_faiss_index(self.pool2_embeddings)

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

    def get_dynamic_few_shot_examples(self, k, pool_1_percentage=1.0):
        """
        Find top k similar questions from the few-shot pool for each question
        """
        k1 = int(round(k * pool_1_percentage))
        k2 = k - k1
        
        print(f"Finding top {k} examples per test question ({k1} from pool 1, {k2} from pool 2)...")
        
        few_shot_map = {}
        for i in tqdm(range(len(self.dataset)), desc="Mapping few-shot examples"):
            test_item = self.dataset[i]
            test_embedding = self.data_embeddings[i]
            
            num_to_search = k + 20  # Oversampling to deal with duplicate ids

            p1_candidates = []
            if k1 > 0 and self.pool1_index:
                p1_dists, p1_indices = self.pool1_index.search(np.array([test_embedding]).astype('float32'), num_to_search)
                p1_candidates = [(p1_dists[0][j], self.pool1_data[p1_indices[0][j]], 'pool1') for j in range(len(p1_dists[0]))]

            p2_candidates = []
            if k2 > 0 and self.pool2_index:
                p2_dists, p2_indices = self.pool2_index.search(np.array([test_embedding]).astype('float32'), num_to_search)
                p2_candidates = [(p2_dists[0][j], self.pool2_data[p2_indices[0][j]], 'pool2') for j in range(len(p2_dists[0]))]
            
            best_candidates = {}
            for dist, example, pool_name in p1_candidates + p2_candidates:
                example_id = example['id']
                if example_id not in best_candidates or dist < best_candidates[example_id][0]:
                    best_candidates[example_id] = (dist, example, pool_name)
            
            best_from_pool1 = sorted([ex for _, ex, pool in best_candidates.values() if pool == 'pool1'], key=lambda x: x['id'])
            best_from_pool2 = sorted([ex for _, ex, pool in best_candidates.values() if pool == 'pool2'], key=lambda x: x['id'])

            # Take the top k1 and k2 from de-dup sorted lists
            final_pool1_examples = best_from_pool1[:k1]
            final_pool2_examples = best_from_pool2[:k2]

            combined_examples = final_pool1_examples + final_pool2_examples
            random.shuffle(combined_examples)
            
            few_shot_map[test_item['id']] = combined_examples
            
        return few_shot_map
