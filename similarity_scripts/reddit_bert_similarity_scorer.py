import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("reddit_data_with_similarity.csv")

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_bert_similarity(text1, text2):
    try:
        embeddings = model.encode([str(text1), str(text2)], convert_to_tensor=True)
        sim = cosine_similarity([embeddings[0].cpu().numpy()], [embeddings[1].cpu().numpy()])[0][0]
        return round(sim * 10, 2)  # Scale from 0 to 10
    except Exception as e:
        print(f"Error computing BERT similarity: {e}")
        return None

df["Expert/MedLLM Response Similarity (BERT)"] = df.apply(
    lambda row: compute_bert_similarity(row["physician_comments"], row["Medical LLM Response"]),
    axis=1
)

df["Expert/General LLM Response Similarity (BERT)"] = df.apply(
    lambda row: compute_bert_similarity(row["physician_comments"], row["General LLM Response"]),
    axis=1
)

df.to_csv("reddit_data_with_all_similarity.csv", index=False)
print("BERT similarity scores saved to 'medical_expert_data_with_all_similarity.csv'")