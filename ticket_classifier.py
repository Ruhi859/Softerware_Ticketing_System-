import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os

# --- Global variables ---
MODEL = None
VECTORIZER = None

# --- ACTION REQUIRED: Paste your personal Gemini API key here ---
# You can get a free key from Google AI Studio.
API_KEY = "AIzaSyBLOzRXhyxgw9d4A0tNuHntlwB4wVNh0g4"

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

# Static solutions are now only used for simple L1 tickets
SOLUTIONS_MAP = {
    'L1': (
        'This is a standard user request. Advised actions: '
        '1. Guide the user through basic troubleshooting (e.g., restart device, check connections). '
        '2. If it is a password reset, follow the standard identity verification protocol. '
        '3. For software requests, confirm user eligibility and create an installation job.'
    )
}

# --- 1. Generative AI Agent for Complex Tickets ---

def get_agent_solution(query):
    """
    Calls the Gemini API to act as an L2/L3 agent, analyzing and resolving complex tickets.
    Returns a dictionary containing the parsed level and the full solution.
    """
    print(f"Escalating to generative agent for query: {query}")
    
    prompt = f"""
    Analyze the following IT support query: "{query}"

    Strictly classify the query as either L2 (High Priority) or L3 (Critical). L3 is reserved for a total service outage.

    Provide a single sentence response that includes ONLY the classification (L2 or L3) and the immediate next step.
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    headers = {
        'Content-Type': 'application/json',
        'x-goog-api-key': API_KEY
    }
    
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        
        result = response.json()
        
        candidate = result.get('candidates', [{}])[0]
        content_part = candidate.get('content', {}).get('parts', [{}])[0]
        generated_text = content_part.get('text', 'Error: Could not parse the AI model response.')
        
        parsed_level = "Agent Analyzed"
        match = re.search(r'\b(L2|L3)\b', generated_text, re.IGNORECASE)
        if match:
            parsed_level = match.group(1).upper()
        
        return {"level": parsed_level, "solution": generated_text.strip()}

    except requests.exceptions.HTTPError as e:
        error_details = f"HTTP Error: {e.response.status_code}. Response: {e.response.text}"
        print(f"Error calling Gemini API: {error_details}")
        return {"level": "Agent API Error", "solution": f"The AI agent returned an error. This is likely an API key or permission issue. Status code: {e.response.status_code}"}
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return {"level": "Agent Connection Error", "solution": "Could not connect to the generative AI agent. Please check the network connection and server logs."}
    except (KeyError, IndexError) as e:
        print(f"Error parsing Gemini API response: {e}")
        print(f"Full response: {response.text if 'response' in locals() else 'No response object'}")
        return {"level": "Agent Response Error", "solution": "Received an unexpected response from the generative AI agent. Please check the server logs."}


# --- 2. Data Loading and Preprocessing ---

def preprocess_text(text):
    """Cleans and preprocesses the input text."""
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I | re.A)
    text = text.lower()
    text = text.strip()
    return text

# --- 3. Model Training ---

def train_global_model():
    """Loads data and trains the local L1 vs. COMPLEX classification model."""
    global MODEL, VECTORIZER
    
    print("Loading data and training L1 vs. COMPLEX triage model...")
    df = pd.read_csv('sample_tickets.csv')
    df['cleaned_query'] = df['query'].apply(preprocess_text)

    X = df['cleaned_query']
    y = df['level']
    
    VECTORIZER = TfidfVectorizer(stop_words='english', max_features=1000)
    X_tfidf = VECTORIZER.fit_transform(X)

    MODEL = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    MODEL.fit(X_tfidf, y)
    print("Triage model training complete.")

# --- 4. Flask App Setup ---

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to classify a ticket and get a solution."""
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Missing 'query' in request"}), 400

    cleaned_query = preprocess_text(query)
    query_tfidf = VECTORIZER.transform([cleaned_query])
    prediction = MODEL.predict(query_tfidf)
    level = prediction[0]
    
    solution = ""
    level_for_ui = ""

    if level == 'COMPLEX':
        agent_response = get_agent_solution(query)
        solution = agent_response["solution"]
        level_for_ui = agent_response["level"]
    else:
        solution = SOLUTIONS_MAP.get('L1', "No specific solution available.")
        level_for_ui = 'L1'
    
    response = {
        "query": query,
        "level": level_for_ui,
        "solution": solution,
    }
    
    return jsonify(response)

# --- Main Execution ---

if __name__ == "__main__":
    train_global_model()
    app.run(port=5000, debug=True)