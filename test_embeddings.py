import numpy as np
from app.embedder import embed_texts

sentences = [
    "What is the minimum attendance?",
    "How many classes must I attend?",
    "The campus has a library."
]

embeddings = embed_texts(sentences)

print("\n========== EMBEDDING SANITY TEST ==========")
print("Embedding shape:", embeddings.shape)
print("Data type:", embeddings.dtype)

# Calculate cosine similarities.
# Because embeddings are normalized, their dot products
# are equal to their cosine similarities.

similarity_1_2 = np.dot(embeddings[0], embeddings[1])
similarity_1_3 = np.dot(embeddings[0], embeddings[2])
similarity_2_3 = np.dot(embeddings[1], embeddings[2])

print("\n========== SIMILARITY SCORES ==========")
print(
    "Minimum attendance vs classes to attend:",
    round(float(similarity_1_2), 4)
)
print(
    "Minimum attendance vs library:",
    round(float(similarity_1_3), 4)
)
print(
    "Classes to attend vs library:",
    round(float(similarity_2_3), 4)
)

if similarity_1_2 > similarity_1_3 and similarity_1_2 > similarity_2_3:
    print("\nPASS: Related attendance questions are more similar.")
else:
    print("\nCHECK: The expected similarity relationship was not observed.")