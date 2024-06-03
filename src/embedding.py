
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_CACHE'] = './cache'
os.environ['HF_HOME'] = './cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
model_path = './models/gte-base-en-v1.5'

from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'trust_remote_code': True},
        multi_process=False
    )
