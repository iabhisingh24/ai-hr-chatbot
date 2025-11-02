import requests
import json

print("=== Testing /ingest Endpoint ===")

# Test 1: Re-index existing documents (no file upload)
print("\n1. Testing re-index existing documents...")
response = requests.post("http://localhost:8000/ingest")
print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=2))

# Test 2: Test with a sample text file (optional)
print("\n2. To test file upload, run this command in terminal:")
print('curl -X POST "http://localhost:8000/ingest" -F "file=@docs/hr_policy.txt"')

# Test 3: Verify the system still works after re-indexing
print("\n3. Testing /ask after re-indexing...")
ask_response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What are working hours?", "top_k": 2}
)
print("Ask Status:", ask_response.status_code)
if ask_response.status_code == 200:
    print("✅ System working after re-indexing!")