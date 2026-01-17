import os
import re
import requests
import pandas as pd
from fastapi import HTTPException

class CivicClassifier:
    def __init__(self):
            # Load CSVs
            self.depts_df = pd.read_csv('tag_departments.csv')
            self.keywords_df = pd.read_csv('tag_keywords.csv')
    
            self.tag_to_dept = {
                str(k).lower().strip(): str(v).strip() 
                for k, v in zip(self.depts_df['Tag'], self.depts_df['Department'])
            }
            # Lowercase the keywords for matching
            self.keywords_df['Keyword'] = self.keywords_df['Keyword'].astype(str).str.lower().str.strip()
            self.keywords_df['Tag'] = self.keywords_df['Tag'].astype(str).str.lower().str.strip()
            
            self.categories = [str(cat).lower().strip() for cat in self.depts_df['Tag'].unique()]
            
            #Setup Hugging Face Router Configuration
            self.api_url = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
            self.token = os.getenv("HF_TOKEN")
    
            self.headers = {"Authorization": f"Bearer {self.token}"}
    
    def tag_issue(self, description: str):
        desc_lower = description.lower()

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
            #multiple tags use one which comes first
            first_match = sorted(matches, key=lambda x: x['pos'])[0]
            tag = first_match['tag']
            return {
                "tag": tag,
                "department": self.tag_to_dept.get(tag, "General Grievances Cell /Public Relations Office"),
                "confidence": 1.0,
                "method": "keyword_match"
            }
        print('moving to AI');
       #if not found in csb file use model
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
            

            top_prediction = result[0]
            predicted_tag = top_prediction['label']
            confidence = top_prediction['score']

            return {
                "tag": predicted_tag,
                "department": self.tag_to_dept.get(predicted_tag, "General Grievances Cell /Public Relations Office")
                "confidence": round(confidence, 2),
                "method": "transformer_model"
            }

        except Exception as e:
            # Log the specific error
            print(f"AI Inference Failure: {e}")
            raise e
