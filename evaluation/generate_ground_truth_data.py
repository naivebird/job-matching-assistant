import pandas as pd
import tqdm

from ingestion.ingestion import load_documents
from rag.rag import get_openai_client


def generate_resume(job_description):
    prompt_template = """
    Based on the following job description, generate an one-page resume that would be a strong match for the role.
    If you can, don't just copy exactly the job description.
    Instead, create a resume that highlights relevant skills and experiences.

    Job Description:
    {job_description}
    """.strip()

    prompt = prompt_template.format(job_description=job_description)
    response = get_openai_client().chat.completions.create(
        model='gpt-4.1-nano',
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content


def generate_ground_truth_data(file_path, num_samples):
    results = []
    documents = load_documents(file_path)
    for doc in tqdm.tqdm(documents[:num_samples]):
        job_description = doc["clean_description"]
        resume = generate_resume(job_description)
        results.append([doc["id"], resume, job_description])
    df = pd.DataFrame(results, columns=["job_id", "generated_resume", "job_description"])
    df.to_csv("data/ground-truth-data.csv", index=False)


if __name__ == '__main__':
    generate_ground_truth_data(
        file_path="data/ground-truth-data.csv",
        num_samples=100
    )
