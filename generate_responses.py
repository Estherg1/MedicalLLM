import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_csv("medical_expert_data-complete.csv")

if 'Question' not in df.columns:
    raise ValueError("CSV must contain a 'question' column.")

def get_llm_response(question):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": question}],
            max_tokens=400,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

df['General LLM Response'] = df['Question'].apply(get_llm_response)

df.to_csv("medical_expert_data_with_responses.csv", index=False)

print("All done! Responses saved to medical_expert_data_with_responses.csv")
