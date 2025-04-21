import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the CSV file
df = pd.read_csv("medical_expert_data-complete.csv").head(5)

# Check if the 'question' column exists
if 'Question' not in df.columns:
    raise ValueError("CSV must contain a 'question' column.")

# Function to generate a response for a given question
def get_llm_response(question):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": question}],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Apply the function to the questions
df['General LLM Response'] = df['Question'].apply(get_llm_response)

# Save the new CSV
df.to_csv("medical_expert_data_with_responses.csv", index=False)

print("All done! Responses saved to medical_expert_data_with_responses.csv")
