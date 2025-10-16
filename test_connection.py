import requests
import json

# --- 1. PASTE YOUR GEMINI API KEY HERE ---
API_KEY = "AIzaSyDISQfGaq8K2a3DHOxPSU-N9RRn41Q1XNg"

# --- 2. Define the API URL ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

def run_connection_test():
    """Runs a simple test to check for network connectivity issues."""
    
    # --- Test 1: Can we reach a general Google server? ---
    print("--- Running Test 1: Connecting to www.google.com ---")
    try:
        response = requests.get("https://www.google.com", timeout=10)
        response.raise_for_status()
        print("✅ SUCCESS: Successfully connected to Google's main server.\n")
    except requests.exceptions.RequestException as e:
        print(f"❌ FAILURE: Could not connect to Google's main server.")
        print(f"   Error: {e}")
        print("   This indicates a general network problem (firewall, no internet, etc.).\n")
        return # Stop if this basic test fails

    # --- Test 2: Can we reach the Gemini API server? ---
    print("--- Running Test 2: Connecting to the Gemini API ---")
    headers = {
        'Content-Type': 'application/json',
        'x-goog-api-key': API_KEY
    }
    payload = {
        "contents": [{"parts": [{"text": "hello"}]}]
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        print("✅ SUCCESS: Successfully connected to the Gemini API server.")
        print("   Your network and API key are working correctly!")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ FAILURE: Could not connect to the Gemini API server.")
        print(f"   Error: {e}")
        print("   This strongly suggests a firewall or network policy is blocking this specific connection.")

if __name__ == "__main__":
    if "PASTE_YOUR" in API_KEY:
        print("⚠️ Please paste your API key into the script before running.")
    else:
        run_connection_test()