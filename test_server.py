from requests import post

endpoint = 'http://127.0.0.1:5000/generate'
example = {"text":"this is a test"}

response = post(endpoint, json=example)
print(response)
if response.ok:
    print(response.json())


