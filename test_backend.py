#!/usr/bin/env python3
import requests
import sys

BASE_URL = "http://localhost:8000"

# List of endpoints to test
endpoints = [
    "/api/public/buses",
    "/api/public/branches",
    "/api/public/generators",
    "/api/public/loads",
    "/api/public/substations",
    "/api/public/bas",
    "/api/public/heatmap?parameter=temperature&date=2020-07-21"
]

def test_endpoints():
    print(f"Testing backend API at {BASE_URL}")
    print("-" * 50)
    
    all_success = True
    
    for endpoint in endpoints:
        url = f"{BASE_URL}{endpoint}"
        try:
            print(f"Testing: {url}")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    print(f"✅ Success! Received {len(data)} items")
                else:
                    print(f"✅ Success! Response: {response.status_code}")
            else:
                print(f"❌ Failed with status code: {response.status_code}")
                print(f"Response: {response.text[:200]}")
                all_success = False
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            all_success = False
            
        print("-" * 50)
    
    return all_success

if __name__ == "__main__":
    success = test_endpoints()
    if not success:
        print("⚠️ Some endpoints failed. Backend may need initialization.")
        sys.exit(1)
    else:
        print("🎉 All endpoints are working!")
        sys.exit(0) 