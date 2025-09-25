import os
from functools import cache

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

OPENAI_MODEL_VERSION = "gpt-5"

load_dotenv()


@cache
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


@cache
def get_es_client():
    return Elasticsearch('http://elasticsearch:9200')


@cache
def get_openai_client():
    return OpenAI()


def load_resume(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def hybrid_search(resume, es_client=None, index_name="seek-jobs", top_k=5):
    resume_embedding = load_embedding_model().encode([resume])[0]
    query = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {"match": {"description": resume}},
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'description_vector') + 1.0",
                                "params": {"query_vector": resume_embedding}
                            }
                        }
                    }
                ]
            }
        },
        "_source": ["id", "title", "description", "company.name", "location.where", "url"]
    }
    response = (es_client or get_es_client()).search(index=index_name, body=query)
    return response['hits']['hits']


def build_prompt(resume_text, jobs, return_job_desc_only=False):
    prompt = f"Based on the following resume:\n{resume_text}\n\n"
    prompt += "Here are some job descriptions that match the resume:\n"
    for i, job in enumerate(jobs):
        source = job['_source']
        prompt += f"{i + 1}. {source['title']} at {source['company']['name']} in {source['location']['where']}\n"
        prompt += f"   Job Description: {source['description']}\n"
        prompt += f"   URL: {source['url']}\n\n"
    if return_job_desc_only:
        prompt += "Return the best-match job description in text format"
    else:
        prompt += "Identify the best match, return its title and URL, explain why the resume matches and which required skills are missing."
    return prompt


def llm(prompt, openai_model_version):
    response = get_openai_client().chat.completions.create(
        model=openai_model_version or OPENAI_MODEL_VERSION,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def rag(resume, return_job_desc_only=False, es_client=None, openai_model_version=None, top_k=5):
    top_jobs = hybrid_search(
        es_client=es_client,
        resume=resume,
        top_k=top_k
    )
    prompt = build_prompt(resume, top_jobs, return_job_desc_only)

    return llm(prompt, openai_model_version)


if __name__ == '__main__':
    resume_file = 'data/Technical Resume (Data Science).pdf'
    resume = load_resume(resume_file)

    result = rag(resume=resume,
                 top_k=5)
    print(result)
