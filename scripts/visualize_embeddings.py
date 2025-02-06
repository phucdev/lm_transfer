import argparse
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


# Get top-k nearest neighbors for each token
def get_top_neighbors(similarities, token_id, tokenizer, k=5, pool_size=20, diversify=False):
    """
    Get top-k nearest neighbors for a given token ID based on cosine similarities.
    :param similarities: Similarity scores to all words in RoBERTa's vocabulary
    :param token_id: Token ID of the input word
    :param tokenizer: The tokenizer
    :param k: Top-k neighbors to return
    :param pool_size: Top-pool_size neighbors to consider when diversify=True
    :param diversify: Exclude minor variants of the same word
    :return: Token IDs of the top-k nearest neighbors
    """
    sorted_indices = np.argsort(similarities)[::-1]  # Sort descending
    sorted_indices = sorted_indices[sorted_indices != token_id]  # Remove the input token
    if diversify:
        candidate_indices = sorted_indices[:pool_size]  # Consider top 20 first

        unique_words = {tokenizer.convert_ids_to_tokens(token_id).strip().replace("Ġ", "").lower()}
        selected_neighbors = []

        for idx in candidate_indices:
            word = tokenizer.convert_ids_to_tokens(int(idx)).strip().replace("Ġ", "")  # Remove leading spaces
            word_lower = word.lower()  # Normalize case

            # Check if it's a duplicate or minor variant
            if word_lower not in unique_words:
                unique_words.add(word_lower)
                selected_neighbors.append(idx)

            # Stop if we found enough diverse words
            if len(selected_neighbors) >= k:
                break
    else:
        selected_neighbors = sorted_indices[:k]

    return selected_neighbors


def plot_embeddings(embeddings, tokenizer, token_id_a, token_id_b, output_file_path, diversify=False):
    a_emb = embeddings[token_id_a]
    b_emb = embeddings[token_id_b]

    # Compute cosine similarity to all words in RoBERTa's vocabulary
    cos_sim = cosine_similarity([a_emb, b_emb], embeddings)

    top_k = 5
    a_neighbors = get_top_neighbors(cos_sim[0], token_id_a, tokenizer, k=top_k, diversify=diversify)
    b_neighbors = get_top_neighbors(cos_sim[1], token_id_b, tokenizer, k=top_k, diversify=diversify)

    # Get corresponding words
    a_words = tokenizer.convert_ids_to_tokens(a_neighbors)
    b_words = tokenizer.convert_ids_to_tokens(b_neighbors)

    # Stack embeddings for visualization
    embedding_subset = np.vstack([
        a_emb, embeddings[a_neighbors], b_emb, embeddings[b_neighbors]
    ])

    # Reduce dimensions using t-SNE
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = umap_reducer.fit_transform(embedding_subset)

    # Extract x, y positions
    x, y = embedding_2d[:, 0], embedding_2d[:, 1]

    # Plot
    plt.figure(figsize=(8, 6))

    token_a = tokenizer.convert_ids_to_tokens(token_id_a)
    token_b = tokenizer.convert_ids_to_tokens(token_id_b)

    # Plot "tre" and its neighbors
    plt.scatter(x[0], y[0], color="red", label=token_a)
    for i in range(1, top_k + 1):
        plt.scatter(x[i], y[i], color="lightcoral")
        plt.text(x[i], y[i], a_words[i - 1], fontsize=9, color="darkred")

    # Plot "bamboo" and its neighbors
    plt.scatter(x[top_k + 1], y[top_k + 1], color="blue", label=token_b)
    for i in range(top_k + 2, len(x)):
        plt.scatter(x[i], y[i], color="lightblue")
        plt.text(x[i], y[i], b_words[i - (top_k + 2)], fontsize=9, color="darkblue")

    plt.title(f"UMAP Projection of '{token_a}' and '{token_b}' in RoBERTa's Embedding Space")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(output_file_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize embeddings in RoBERTa's vocabulary")
    parser.add_argument("--output_file", type=str, help="Path to save the plot")
    return parser.parse_args()


def main():
    args = parse_args()
    output_file_path = args.output_file
    # Load English RoBERTa model and tokenizer
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # Extract embeddings
    with torch.no_grad():
        embeddings = model.embeddings.word_embeddings.weight.cpu().numpy()
    # token IDs for "tre" and "bamboo" = [6110, 31970]
    plot_embeddings(
        embeddings=embeddings,
        tokenizer=tokenizer,
        token_id_a=6110,
        token_id_b=31970,
        diversify=False,
        output_file_path=output_file_path
    )


if __name__ == "__main__":
    main()
