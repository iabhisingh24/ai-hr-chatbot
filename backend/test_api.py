import requests
import json

# Test the updated /ask endpoint
print("=== Testing /ask endpoint ===")
response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "How many paid sick leaves?",
        "filters": {"policy": "Leave"},
        "top_k": 3
    }
)

print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=2))

# Test the /docs endpoint with DEBUGGING
print("\n=== Testing /docs endpoint ===")
docs_response = requests.get("http://localhost:8000/documents")
print("Status Code:", docs_response.status_code)
print("Raw Response Text:")
print(docs_response.text)  

# Try to parse JSON only if status is 200
if docs_response.status_code == 200:
    try:
        docs_data = docs_response.json()
        print("Parsed Docs Response:")
        print(json.dumps(docs_data, indent=2))
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
else:
    print(f"❌ HTTP Error: {docs_response.status_code}")