import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_csv("reddit_cleaned_data (1).csv")

if 'title' not in df.columns or 'selftext' not in df.columns:
    raise ValueError("CSV must contain both 'title' and 'selftext' columns.")

def get_llm_response(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": text}],
            max_tokens=400,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

df['combined_text'] = df['title'].fillna('') + "\n\n" + df['selftext'].fillna('')

df['General LLM Response'] = df['combined_text'].apply(get_llm_response)

df.to_csv("reddit_data_with_responses.csv", index=False)

print("Responses saved to reddit_data_with_responses.csv")
