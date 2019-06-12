import requests

# Initialize the Keras REST API endpoint URL along with the input
url = "http://localhost:5000/predict"
data = {
  "major_input": "Accounting",
  "degree_type_input": "Bachelor's",
  "managed_others_input": "No",
  "years_exp_input": 2,
  "k": 50}

# Submit the request
response = requests.post(url, headers={'Content-Type': 'application/json'}, json=data).json()
print(response)

