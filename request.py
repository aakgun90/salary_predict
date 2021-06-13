import requests
url = "http://127.0.0.1:5000/predict_api"
r = requests.post(url, json= {"exprerience":2, "test_score(out of 10)":9, "interview_score(out of 10)":6})
print(r.json())