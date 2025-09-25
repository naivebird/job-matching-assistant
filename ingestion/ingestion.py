import json
from functools import cache

import tqdm
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


@cache
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


@cache
def get_es_client():
    return Elasticsearch('http://localhost:9200')


def get_mappings():
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "title": {"type": "text"},
                "description": {"type": "text"},
                "description_vector": {
                    "type": "dense_vector",
                    "dims": 384
                },
                "types": {"type": "keyword"},
                "arrangement": {"type": "keyword"},
                "url": {"type": "keyword"},
                "listing_date": {"type": "date"},
                "company": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "name": {"type": "text"}
                    }
                }
            }
        }
    }


def create_index(index_name, mappings):
    es_client = get_es_client()
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
    es_client.indices.create(index=index_name, body=mappings)


def load_documents(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def ingest_data(index_name):
    documents = load_documents("data/seek_jobs.json")
    for doc in tqdm.tqdm(documents):
        get_es_client().index(
            index=index_name,
            document={
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["clean_description"],
                "description_vector": load_embedding_model().encode(doc["clean_description"]).tolist(),
                "types": doc["types"],
                "arrangement": doc["arrangement"],
                "url": doc["url"],
                "listing_date": doc["listingDate"],
                "company": {
                    "id": doc["company"]["id"],
                    "name": doc["company"]["name"]
                }
            }
        )


if __name__ == '__main__':
    index_name = "seek-jobs"
    create_index(index_name, get_mappings())
    ingest_data(index_name)
