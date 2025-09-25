import json

import pandas as pd

from rag.rag import get_openai_client


def generate_resume(openai_client, job_description):
    prompt_template = """
    Based on the following job description, generate an one-page resume that would be a strong match for the role.
    If you can, don't just copy exactly the job description.
    Instead, create a resume that highlights relevant skills and experiences.

    Job Description:
    {job_description}
    """.strip()

    prompt = prompt_template.format(job_description=job_description)
    response = openai_client.chat.completions.create(
        model='gpt-4.1-nano',
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content

def generate_ground_truth_data(documents, num_samples):
    openai_client = get_openai_client()
    results = []
    for  doc in documents[:num_samples]:
        job_description = doc["clean_description"]
        resume = generate_resume(openai_client, job_description)
        results.append([doc["id"], resume, job_description])
        print(f"Generated resume for job ID {doc['id']}")
    df = pd.DataFrame(results, columns=["job_id", "generated_resume", "job_description"])
    df.to_csv("data/ground-truth-data.csv", index=False)

if __name__ == '__main__':
    with open("data/seek-jobs.json", "r") as f:
        documents = json.load(f)
    generate_ground_truth_data(documents, num_samples=100)