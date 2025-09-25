import requests

url = "http://localhost:8000/predict"
file_path = "data/Technical Resume (Data Science).pdf"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "application/pdf")}
    response = requests.post(url, files=files).json()

print("Response:", response["result"])
