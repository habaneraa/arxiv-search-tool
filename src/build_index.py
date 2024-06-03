
import json
import time
import jsonlines
import torch
from itertools import islice
from typing import Iterable, Generator, TypeVar
from tqdm import tqdm
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant

from embedding import get_embedding_model
from config import config


T = TypeVar("T")


def generate_batch(data: Iterable[T], batch_size: int = 32) -> Generator[list[T], None, None]:
    iterator = iter(data)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def build_doc_from_dict(paper) -> Document:
    return Document(
        page_content=paper['abstract'],
        metadata={
            "id": paper['id'], 
            "title": paper['title'], 
            "categories": paper['categories'],
            "update_date": datetime.strptime(paper['update_date'], "%Y-%m-%d").date(),
        },
    )


def main():
    # 准备向量数据库集合
    client = QdrantClient(config.qdrant_url)
    client.delete_collection('papers')
    client.create_collection(
        'papers',
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    embedding_model = get_embedding_model()
    qdrant_vector_store = Qdrant(client, 'papers', embedding_model)
    
    # Load arXiv papers
    # URL: https://www.kaggle.com/datasets/Cornell-University/arxiv/data
    dataset_path = config.dataset_path
    all_papers = []
    with jsonlines.open(dataset_path, 'r') as reader:
        for item in tqdm(reader, total=2463961):
            item.pop('versions')
            item.pop('authors_parsed')
            has_target_category = False
            for c in item['categories'].split():
                if c.startswith('cs'):
                    has_target_category = True
                    break
            if has_target_category:
                all_papers.append(item)
    print(f'Loaded {len(all_papers)} papers!')

    documents = [build_doc_from_dict(p) for p in all_papers]

    batch_size = 64
    num_batches = len(documents) // batch_size + 1
    with torch.autocast(device_type=torch.device('cuda').type, dtype=torch.float16):
        with torch.inference_mode():
            for batch in tqdm(generate_batch(documents, batch_size), total=num_batches):
                ret = qdrant_vector_store.add_documents(batch)
    
    print('Completed!')


if __name__ == '__main__':
    main()
