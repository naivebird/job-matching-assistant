import pandas as pd

from ingestion.ingestion import get_es_client, load_embedding_model
from rag.rag import hybrid_search


def dense_vector_search(resume, index_name="seek-jobs", top_k=5):
    resume_embedding = load_embedding_model().encode([resume])[0]
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'description_vector') + 1.0",
                "params": {"query_vector": resume_embedding}
            }
        }
    }

    response = get_es_client().search(
        index=index_name,
        body={
            "size": top_k,
            "query": script_query,
            "_source": ["id", "title", "description", "company.name", "location.where", "url"]
        }
    )

    results = response['hits']['hits']
    return results


def local_hybrid_search(**kwargs):
    return hybrid_search(es_client=get_es_client(), **kwargs)


def compute_relevance_matrix(search_func, ground_truth_df):
    relevance_matrix = []
    for _, row in ground_truth_df.iterrows():
        job_id = str(row['job_id'])
        resume = row['generated_resume']
        search_results = search_func(resume=resume, top_k=5)
        relevance_array = [job['_source']['id'] == job_id for job in search_results]
        relevance_matrix.append(relevance_array)
    return relevance_matrix


def compute_hit_rate(relevance_matrix):
    hits = sum(1 for relevance in relevance_matrix if any(relevance))
    return hits / len(relevance_matrix) if relevance_matrix else 0.0


def compute_mrr(relevance_matrix):
    total_score = 0.0
    for relevance in relevance_matrix:
        for rank, is_relevant in enumerate(relevance, start=1):
            if is_relevant:
                total_score += 1.0 / rank
                break
    return total_score / len(relevance_matrix) if relevance_matrix else 0.0


def evaluate_search_methods(ground_truth_file):
    ground_truth_df = pd.read_csv(ground_truth_file)

    relevance_matrix_cosine = compute_relevance_matrix(dense_vector_search, ground_truth_df)
    relevance_matrix_hybrid = compute_relevance_matrix(local_hybrid_search, ground_truth_df)

    hit_rate_cosine = compute_hit_rate(relevance_matrix_cosine)
    mrr_cosine = compute_mrr(relevance_matrix_cosine)

    hit_rate_hybrid = compute_hit_rate(relevance_matrix_hybrid)
    mrr_hybrid = compute_mrr(relevance_matrix_hybrid)

    print(f"Dense Vector Search - Hit Rate: {hit_rate_cosine:.4f}, MRR: {mrr_cosine:.4f}")
    print(f"Hybrid Search - Hit Rate: {hit_rate_hybrid:.4f}, MRR: {mrr_hybrid:.4f}")


if __name__ == '__main__':
    evaluate_search_methods("data/ground-truth-data.csv")
