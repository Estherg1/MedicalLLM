import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_csv("medical_expert_data_with_responses.csv")

def get_similarity_score(expert_answer, medllm_answer):
    try:
        prompt = (
            f"Rate the conceptual similarity between the following two answers on a scale from 0 to 10, "
            f"where 10 means they are conceptually identical and 0 means they are completely unrelated.\n\n"
            f"Expert Answer:\n{expert_answer}\n\n"
            f"MedLLM Answer:\n{medllm_answer}\n\n"
            f"Score (just a number from 0 to 10):"
        )

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )

        content = response.choices[0].message.content.strip()
        score = float(content) if content.replace(".", "", 1).isdigit() else None
        return score

    except Exception as e:
        print(f"Error comparing answers: {e}")
        return None

df["Expert/MedLLM Response Similarity"] = df.apply(
    lambda row: get_similarity_score(row["Answer"], row["MedLLM Answer"]),
    axis=1
)

df["Expert/General LLM Response Similarity"] = df.apply(
    lambda row: get_similarity_score(row["Answer"], row["General LLM Response"]),
    axis=1
)

df.to_csv("medical_expert_data_with_similarity.csv", index=False)
print("Done! File with similarity scores saved as 'medical_expert_data_with_similarity.csv'")
