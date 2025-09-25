import pandas as pd

from rag.rag import rag, load_embedding_model
from ingestion.ingestion import get_es_client


def compute_similarity(record, response_col):
    answer_orig = record['job_description']
    answer_llm = record[response_col]

    v_llm = load_embedding_model().encode(answer_llm)
    v_orig = load_embedding_model().encode(answer_orig)

    return v_llm.dot(v_orig)


def evaluate_rag(ground_truth_file):
    df = pd.read_csv(ground_truth_file)
    similarity_columns = []
    for llm_model in ['gpt-4o', 'gpt-4.1', 'gpt-5']:
        df[f"{llm_model}_response"] = df.apply(lambda row: rag(resume=row["generated_resume"],
                                                               es_client=get_es_client(),
                                                               return_job_desc_only=True), axis=1)
        df[f"{llm_model}_similarity"] = df.apply(lambda row: compute_similarity(record=row,
                                                                                response_col=f"{llm_model}_response"),
                                                 axis=1)
        similarity_columns.append(f"{llm_model}_similarity")
    print(df[similarity_columns].describe())


if __name__ == '__main__':
    evaluate_rag("data/ground-truth-data.csv")
