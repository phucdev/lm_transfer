import argparse
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
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


def plot_embeddings(
        embeddings,
        tokenizer,
        token_ids,
        output_file_path,
        diversify=False,
        top_k=5,
        dimension_reduction="umap"
):
    """
    Plot the embeddings of tokens and their top-k nearest neighbors in embedding space.
    :param embeddings: The word embeddings
    :param tokenizer: The tokenizer
    :param token_ids: The token IDs of the input words
    :param output_file_path: The path to save the plot
    :param diversify: Whether to exclude minor variants of the same word
    :param top_k: The number of nearest neighbors to consider
    :param dimension_reduction: The dimensionality reduction technique to use ("umap" or "tsne")
    """
    all_embeddings = []
    all_words = []
    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink"]  # Different colors for different tokens
    color_map = {}

    # Compute cosine similarities and find nearest neighbors
    for idx, token_id in enumerate(token_ids):
        token_emb = embeddings[token_id]
        cos_sim = cosine_similarity([token_emb], embeddings)[0]

        # Get top-k diverse neighbors (assumed function)
        neighbors = get_top_neighbors(cos_sim, token_id, tokenizer, k=top_k, diversify=diversify)
        neighbor_words = tokenizer.convert_ids_to_tokens(neighbors)

        # Store embeddings
        all_embeddings.append(token_emb)
        all_embeddings.extend(embeddings[neighbors])

        # Store words
        token_word = tokenizer.convert_ids_to_tokens([token_id])[0]
        all_words.append((token_word, colors[idx % len(colors)], True))  # True -> Main token
        for neighbor_word in neighbor_words:
            all_words.append((neighbor_word, colors[idx % len(colors)], False))  # False -> Neighbor

        # Store color mapping for labels
        color_map[token_word] = colors[idx % len(colors)]

    # Convert embeddings list to numpy array
    all_embeddings = np.vstack(all_embeddings)

    # Reduce dimensions
    if dimension_reduction == "umap":
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = umap_reducer.fit_transform(all_embeddings)
    elif dimension_reduction == "tsne":
        tsne = TSNE(n_components=2, random_state=42)
        embedding_2d = tsne.fit_transform(all_embeddings)
    else:
        raise ValueError("Invalid dimension_reduction argument. Choose either 'umap' or 'tsne'.")

    # Extract x, y positions
    x, y = embedding_2d[:, 0], embedding_2d[:, 1]

    # Plot
    plt.figure(figsize=(10, 7))

    for i, (word, color, is_main) in enumerate(all_words):
        plt.scatter(x[i], y[i], color=color, alpha=1.0 if is_main else 0.6, label=word if is_main else "")
        plt.text(x[i], y[i], word, fontsize=9, color="black")

    plt.title("UMAP Projection of Selected Tokens in Embedding Space")
    plt.legend(loc="best", fontsize=9, markerscale=0.7)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize embeddings in RoBERTa's vocabulary")
    parser.add_argument("--output_file_path", required=True, type=str, help="Path to save the plot")
    return parser.parse_args()


def main():
    args = parse_args()
    output_file_path = args.output_file_path
    # Load English RoBERTa model and tokenizer
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # Extract embeddings
    with torch.no_grad():
        embeddings = model.embeddings.word_embeddings.weight.cpu().numpy()
    # token IDs for " tre" and " bamboo" = [6110, 31970], " con" and " child" = [2764, 920]
    plot_embeddings(
        embeddings=embeddings,
        tokenizer=tokenizer,
        token_ids=[6110, 31970, 2764, 920],
        diversify=False,
        output_file_path=output_file_path
    )


if __name__ == "__main__":
    main()
