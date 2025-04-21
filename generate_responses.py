import pandas as pd
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

file_path = "medical_expert_data-complete.csv"
df = pd.read_csv(file_path)

print(df)

if 'Question' not in df.columns:
    raise ValueError("The CSV file must contain a column named 'question'.")

def get_llm_response(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": question}],
            max_tokens=300,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

df['General LLM Response'] = df['Question'].apply(get_llm_response)

# Save the output
output_file = "all_data.csv"
df.to_csv(output_file, index=False)

print(f"All done! Responses saved to {output_file}")
