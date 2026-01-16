import os
import re
import requests
import pandas as pd
from fastapi import HTTPException

class CivicClassifier:
    def __init__(self):
        # 1. Load mappings from your CSV files
        # Ensure these files are in your project root
        try:
            self.depts_df = pd.read_csv('tag_departments.csv')
            self.keywords_df = pd.read_csv('tag_keywords.csv')
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            raise RuntimeError("Required CSV mapping files are missing.")

        # Create a lookup dictionary: { 'pothole': 'Public Works Department', ... }
        self.tag_to_dept = dict(zip(self.depts_df['Tag'], self.depts_df['Department']))
        
        # Use unique tags as candidate labels for the AI model
        self.categories = self.depts_df['Tag'].unique().tolist()
        
        # 2. Setup Hugging Face Router Configuration
        # Note: Using the updated router URL as per HF infrastructure changes
        self.api_url = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
        self.token = os.getenv("HF_TOKEN")
        
        if not self.token:
            print("Warning: HF_TOKEN not found in environment variables.")

        self.headers = {"Authorization": f"Bearer {self.token}"}

    def tag_issue(self, description: str):
        desc_lower = description.lower()

        # --- STEP 1: KEYWORD & HINGLISH CHECK ---
        # We use regex to ensure we match whole words (e.g., 'pani' matches 'pani leak' but not 'panini')
        matches = []
        for _, row in self.keywords_df.iterrows():
            keyword = str(row['Keyword']).lower()
            if re.search(r'\b' + re.escape(keyword) + r'\b', desc_lower):
                predicted_tag = row['Tag']
                matches.append({
                    "tag": predicted_tag,
                    "pos": desc_lower.find(keyword)
                })

        if matches:
            # If multiple keywords match, prioritize the one mentioned first in the sentence
            first_match = sorted(matches, key=lambda x: x['pos'])[0]
            tag = first_match['tag']
            return {
                "tag": tag,
                "department": self.tag_to_dept.get(tag, "General Administration"),
                "confidence": 1.0,
                "method": "keyword_match"
            }

        # --- STEP 2: AI FALLBACK (Hugging Face Router) ---
        payload = {
            "inputs": description,
            "parameters": {"candidate_labels": self.categories},
            "options": {"wait_for_model": True}
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=10)
            
            if response.status_code != 200:
                # Handle cases like 503 (Loading) or 401 (Unauthorized)
                error_detail = response.json().get("error", "Unknown API Error")
                raise Exception(f"HF Router Error {response.status_code}: {error_detail}")

            result = response.json()
            
            # The router returns a list: [{'label': 'tag_name', 'score': 0.99}, ...]
            # We take the top result
            top_prediction = result[0]
            predicted_tag = top_prediction['label']
            confidence = top_prediction['score']

            return {
                "tag": predicted_tag,
                "department": self.tag_to_dept.get(predicted_tag, "General Administration"),
                "confidence": round(confidence, 2),
                "method": "transformer_model"
            }

        except Exception as e:
            # Log the specific error for debugging during the hackathon
            print(f"AI Inference Failure: {e}")
            raise e