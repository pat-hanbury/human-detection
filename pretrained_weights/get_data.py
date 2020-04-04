import requests

api_key = "-v6idL2.EZLBpCZGBrjDS7PCQLXgTMWC0opHuGbg"
# api_key = "xxx"

json = {"name": "jan-21"}

headers = {"Authorization": f"ApiKey {api_key}"}
url = "https://darwin.v7labs.com/api/datasets"

result = requests.post(url, json=json, headers=headers)
print(result.json())
