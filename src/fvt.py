"""
Code adapted from https://github.com/LeonidasY/fast-vocabulary-transfer/blob/main/fvt/
From the paper:
@inproceedings{gee-etal-2022-fast,
    title = "Fast Vocabulary Transfer for Language Model Compression",
    author = "Gee, Leonidas  and
      Zugarini, Andrea  and
      Rigutini, Leonardo  and
      Torroni, Paolo",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-industry.41",
    pages = "409--416",
}
"""

import re
import torch
import abc
import argparse
import transformers
import logging
import numpy as np

from focus import get_overlapping_tokens

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AbstractVocabularyTransfer(metaclass=abc.ABCMeta):

    def __init__(self):
        self.tokens_map = None

    @staticmethod
    @abc.abstractmethod
    def train_tokenizer(data, source_tokenizer, vocab_size, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def tokens_mapping(self, target_tokenizer, source_tokenizer, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def embeddings_assignment(self, tokens_map, source_model, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def update_model_embeddings(self, source_model, in_vocab, in_matrix, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def transfer(self, in_domain_data, source_tokenizer, source_model, vocab_size, **kwargs):
        raise NotImplementedError


class VocabularyTransfer(AbstractVocabularyTransfer):

    def __init__(self):
        super().__init__()

    @staticmethod
    def train_tokenizer(data, source_tokenizer, vocab_size, **kwargs):
        """
        Train an HF tokenizer with the specified vocab size.

        :param data: a list of textual sequences to train the tokenizer with
        :param source_tokenizer: a general-purpose tokenizer.
        :param vocab_size: int. Vocabulary size for the new trained tokenizer
        :param kwargs: no kwargs

        :return: A new trained tokenizer in the in-domain data
        """

        target_tokenizer = source_tokenizer.train_new_from_iterator(data, vocab_size)

        return target_tokenizer

    @abc.abstractmethod
    def embeddings_assignment(self, tokens_map, source_model, **kwargs):
        raise NotImplementedError

    def update_model_embeddings(self, source_model, in_matrix, target_tokenizer, **kwargs):
        """
        Method that replaces the embeddings of a given LM with in_matrix.

        :param source_model: An huggingface model, e.g. bert
        :param in_matrix: (2-d np.ndarray) The new embedding matrix.
        :param target_tokenizer: Any huggingface tokenizer
        :param kwargs: no kwargs

        :return: A new LM model with replaced embeddings
        """

        # Change the model's embedding matrix
        target_model = source_model
        target_model.resize_token_embeddings(len(target_tokenizer))
        target_model.get_input_embeddings().weight.data = torch.from_numpy(in_matrix)

        tie_weights = kwargs.get('tie_weights', True)
        if tie_weights:
            # Tie the model's weights
            target_model.tie_weights()

        return target_model

    def transfer(self, target_tokenizer, source_tokenizer, source_model, **kwargs):
        """
        Method that returns a new LM model with transferred embeddings.

        :param target_tokenizer: Any huggingface tokenizer
        :param source_tokenizer: Any huggingface tokenizer
        :param source_model: Any huggingface model
        :param kwargs: no kwargs

        :return: A new in_domain model
        """

        self.tokens_map = self.tokens_mapping(target_tokenizer, source_tokenizer)
        in_matrix = self.embeddings_assignment(self.tokens_map, source_model)
        target_model = self.update_model_embeddings(source_model, in_matrix, target_tokenizer)

        return target_model


class FastVocabularyTransfer(VocabularyTransfer):

    def __init__(self):
        super().__init__()

    def tokens_mapping(
            self,
            target_tokenizer,
            source_tokenizer,
            fuzzy_match_all=False,
            exact_match_all=True,
            match_symbols=False,
            **kwargs
    ):
        """
        This method establish a mapping between each token of
        the target tokenizer (target_tokenizer) to one or more tokens from
        the source (source_tokenizer) tokenizer.

        :param target_tokenizer: Any huggingface tokenizer
        :param source_tokenizer: Any huggingface tokenizer
        :param fuzzy_match_all: bool. If True, fuzzy matching is used to detect overlapping tokens.
        :param exact_match_all: bool. If True, exact matching is used to detect overlapping tokens.
        :param match_symbols: bool. If True, symbols are matched.
        :param kwargs: no kwargs

        :return: A dictionary, having size of the target_tokenizer vocabulary.
         Each key is the index corresponding to a token in the in-tokenizer.
         Values are lists of indexes to the tokens of source_tokenizer.
        """

        source_vocab = source_tokenizer.get_vocab()
        target_vocab = target_tokenizer.get_vocab()
        ngram_vocab = target_tokenizer.ngram_vocab if hasattr(target_tokenizer, 'ngram_vocab') else {}

        source_tokenizer_special_tokens = source_tokenizer.all_special_tokens
        target_tokenizer_special_tokens = target_tokenizer.all_special_tokens
        bert_to_roberta = {
            '[CLS]': '<s>',
            '[SEP]': '</s>',
            '[PAD]': '<pad>',
            '[UNK]': '<unk>',  # assuming an unk token needs mapping
            '[MASK]': '<mask>'  # assuming a mask token needs mapping
        }
        if (all(t.startswith("[") for t in source_tokenizer_special_tokens) and
                all(t.startswith("<") for t in target_tokenizer_special_tokens)):
            special_tokens_mapping = {v: k for k, v in bert_to_roberta.items()}
        elif (all(t.startswith("<") for t in source_tokenizer_special_tokens) and
              all(t.startswith("[") for t in target_tokenizer_special_tokens)):
            special_tokens_mapping = bert_to_roberta
        else:
            # Identity mapping
            special_tokens_mapping = {t: t for t in source_tokenizer_special_tokens}

        overlapping_tokens, missing_tokens = get_overlapping_tokens(target_tokenizer, source_tokenizer,
                                                                    match_symbols=match_symbols,
                                                                    exact_match_all=exact_match_all,
                                                                    fuzzy_match_all=fuzzy_match_all)
        logger.info(f"Overlapping tokens: {len(overlapping_tokens)=}, {len(missing_tokens)=}")

        tokens_map = {}
        for new_token, new_index in target_vocab.items():
            if new_token in overlapping_tokens:
                # if the same token exists in the old vocabulary, take its embedding
                # retrieve the form of the token in the old vocabulary
                new_token_source = overlapping_tokens[new_token].source
                new_token_source_form = new_token_source[0].native_form
                assert new_token_source_form in source_vocab, f"{new_token_source_form} not in source_vocab"
                old_index = source_vocab[new_token_source_form]
                tokens_map[new_index] = [old_index]
            elif new_token in special_tokens_mapping and special_tokens_mapping[new_token] in source_vocab:
                # if the token is a special token, map it to the corresponding special token
                old_index = source_vocab[special_tokens_mapping[new_token]]
                tokens_map[new_index] = [old_index]
            else:
                # if not, tokenize the new token using the old vocabulary
                new_token = re.sub('^(##|Ġ|▁)', '', new_token)
                if new_token in ngram_vocab:
                    token_partition = source_tokenizer(
                        new_token.split('‗'), is_split_into_words=True, add_special_tokens=False, return_tensors='pt'
                    )["input_ids"][0]
                else:
                    token_partition = source_tokenizer(
                        new_token, add_special_tokens=False, return_tensors='pt'
                    )["input_ids"][0]
                tokens_map[new_index] = token_partition.tolist()

        return tokens_map

    def embeddings_assignment(self, tokens_map, source_model, seed=42, **kwargs):
        """
        Given a mapping between two tokenizers and a general-purpose model
        trained on source_tokenizer, this method produces a new embedding matrix
        assigning embeddings from the source_model.

        :param tokens_map: A mapping between new and old tokens. See tokens_mapping(...)
        :param source_model: A huggingface model, e.g. bert
        :param seed: int. Random seed for reproducibility
        :param kwargs: no kwargs

        :return: (2-d torch.Tensor) An embedding matrix with same size of tokens_map.
        """
        np.random.seed(seed)
        gen_matrix = source_model.get_input_embeddings().weight.detach().numpy()
        in_matrix = np.random.normal(
            np.mean(gen_matrix, axis=0),
            np.std(gen_matrix, axis=0),
            (
                len(tokens_map),
                gen_matrix.shape[1]
            )
        )

        for new_index, old_indices in tokens_map.items():
            old_embedding = np.mean(gen_matrix[old_indices], axis=0)
            in_matrix[new_index] = old_embedding

        return in_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Fast Vocabulary Transfer")
    parser.add_argument(
        "--source_model_name_or_path",
        type=str,
        required=True,
        help="Model name or path of the source model."
    )
    parser.add_argument(
        "--target_tokenizer_name_or_path",
        type=str,
        required=True,
        help="Target tokenizer."
    )
    parser.add_argument(
        "--target_model_path",
        type=str,
        required=True,
        help="Target model save path."
    )
    parser.add_argument(
        "--fuzzy_match_all",
        action="store_true",
        default=False,
        help="Use fuzzy matching."
    )
    parser.add_argument(
        "--exact_match_all",
        action="store_true",
        default=True,
        help="Use exact matching."
    )
    parser.add_argument(
        "--match_symbols",
        action="store_true",
        default=False,
        help="Match symbols."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(args)
    source_tokenizer = transformers.AutoTokenizer.from_pretrained(args.source_model_name_or_path)
    source_model = transformers.AutoModel.from_pretrained(args.source_model_name_or_path)
    target_tokenizer = transformers.AutoTokenizer.from_pretrained(args.target_tokenizer_name_or_path)
    target_model_path = args.target_model_path
    fuzzy_match_all = args.fuzzy_match_all
    exact_match_all = args.exact_match_all
    match_symbols = args.match_symbols

    logger.info("Transferring vocabulary...")
    fast_vocabulary_transfer = FastVocabularyTransfer()
    transferred_model = fast_vocabulary_transfer.transfer(
        target_tokenizer, source_tokenizer, source_model,
        fuzzy_match_all=fuzzy_match_all, match_symbols=match_symbols, exact_match_all=exact_match_all
    )
    
    transferred_model.save_pretrained(target_model_path)
    target_tokenizer.save_pretrained(target_model_path)
    logger.info(f"Model and tokenizer saved to {target_model_path}")


if __name__ == '__main__':
    main()
