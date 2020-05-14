api_key = "AIzaSyAPQTYzY46b9iMRQBpDWaKlVUjmvQm8_T8"


import json
import requests
url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +
    '?key=' + api_key)
data_dict = {
    'comment': {'text': 'what kind of $#it name is foo?'},
    'languages': ['en'],
    'requestedAttributes': {'TOXICITY': {}}
}

response = requests.post(url=url, data=json.dumps(data_dict))
response_dict = json.loads(response.content)
print(json.dumps(response_dict, indent=2))