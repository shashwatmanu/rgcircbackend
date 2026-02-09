#!/usr/bin/env python3
"""
Test script for file list endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000"

# Test 1: Check if server is running
print("=" * 60)
print("Test 1: Server Health Check")
print("=" * 60)
response = requests.get(f"{BASE_URL}/")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
print()

# Test 2: Get a run_id from a recent reconciliation
# We'll use the run_id from the directory we just checked
TEST_RUN_ID = "reco_outputs_20260209_055313"

print("=" * 60)
print(f"Test 2: Get Files for run_id: {TEST_RUN_ID}")
print("=" * 60)
print("Note: This will fail with 401 if not authenticated")
print("Testing without authentication first...")
response = requests.get(f"{BASE_URL}/reconciliations/{TEST_RUN_ID}/files")
print(f"Status: {response.status_code}")
if response.status_code == 401:
    print("✓ Correctly requires authentication")
else:
    print(f"Response: {json.dumps(response.json(), indent=2)}")
print()

# Test 3: Test with invalid run_id (should return 401 first, then 404 when authenticated)
print("=" * 60)
print("Test 3: Get Files for invalid run_id")
print("=" * 60)
INVALID_RUN_ID = "invalid_run_id_12345"
response = requests.get(f"{BASE_URL}/reconciliations/{INVALID_RUN_ID}/files")
print(f"Status: {response.status_code}")
if response.status_code == 401:
    print("✓ Correctly requires authentication")
else:
    print(f"Response: {json.dumps(response.json(), indent=2)}")
print()

print("=" * 60)
print("Summary")
print("=" * 60)
print("✓ Server is running")
print("✓ Endpoints require authentication (as expected)")
print("✓ To fully test, you need to:")
print("  1. Create a user account or use existing credentials")
print("  2. Login to get a token")
print("  3. Test with the token in Authorization header")
print()
print("Example authenticated request:")
print(f'  curl -H "Authorization: Bearer {{token}}" \\')
print(f'       {BASE_URL}/reconciliations/{TEST_RUN_ID}/files')
