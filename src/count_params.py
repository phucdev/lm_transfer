import transformers


def main():
    model_name_or_paths = [
        "FacebookAI/roberta-base",
        "FacebookAI/roberta-large",
        "EleutherAI/pythia-160m-deduped",
        "EleutherAI/pythia-410m-deduped",
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
        "bigscience/bloom-560m"
    ]
    for model_name_or_path in model_name_or_paths:
        if "bert" in model_name_or_path:
            model = transformers.AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
        num_params = model.num_parameters()
        vocab_size = model.config.vocab_size
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        hidden_dim = model.config.hidden_size
        if "bloom" in model_name_or_path:
            num_embedding_params = sum(p.numel() for p in model.base_model.word_embeddings.parameters())
            num_lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
        elif "pythia" in model_name_or_path:
            num_embedding_params = sum(p.numel() for p in model.base_model.embed_in.parameters())
            num_lm_head_params = sum(p.numel() for p in model.embed_out.parameters())
        else:
            num_embedding_params = sum(p.numel() for p in model.base_model.embeddings.parameters())
            num_lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
        num_non_embedding_params = num_params - num_embedding_params
        print(f"{model_name_or_path} has {num_params:,} parameters, "
              f"{num_non_embedding_params:,} non-embedding parameters, "
              f"{num_params-num_lm_head_params:,} parameters without LM head, "
              f"{vocab_size:,} vocab size, {num_layers} layers, "
              f"{num_heads} heads, and {hidden_dim} hidden size")


if __name__ == "__main__":
    main()
