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
import torch.nn as nn


class AbstractVocabularyTransfer(metaclass=abc.ABCMeta):

    def __init__(self):
        self.tokens_map = None

    @staticmethod
    @abc.abstractmethod
    def train_tokenizer(data, gen_tokenizer, vocab_size, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def tokens_mapping(self, in_tokenizer, gen_tokenizer, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def embeddings_assignment(self, tokens_map, gen_model, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def update_model_embeddings(self, gen_model, in_vocab, in_matrix, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def transfer(self, in_domain_data, gen_tokenizer, gen_model, vocab_size, **kwargs):
        raise NotImplementedError


class VocabularyTransfer(AbstractVocabularyTransfer):

    def __init__(self):
        super().__init__()

    @staticmethod
    def train_tokenizer(data, gen_tokenizer, vocab_size, **kwargs):
        """
        Train an HF tokenizer with the specified vocab size.

        :param data: a list of textual sequences to train the tokenizer with
        :param gen_tokenizer: a general-purpose tokenizer.
        :param vocab_size: int. Vocabulary size for the new trained tokenizer
        :param kwargs: no kwargs

        :return: A new trained tokenizer in the in-domain data
        """

        in_tokenizer = gen_tokenizer.train_new_from_iterator(data, vocab_size)

        return in_tokenizer

    @abc.abstractmethod
    def embeddings_assignment(self, tokens_map, gen_model, **kwargs):
        raise NotImplementedError

    def update_model_embeddings(self, gen_model, in_matrix, **kwargs):
        """
        Method that replaces the embeddings of a given LM with in_matrix.

        :param gen_model: An huggingface model, e.g. bert
        :param in_matrix: (2-d torch.Tensor) The new embedding matrix.
        :param kwargs: no kwargs

        :return: A new LM model with replaced embeddings
        """

        # Change the model's embedding matrix
        gen_model.get_input_embeddings().weight = nn.Parameter(in_matrix)
        gen_model.config.vocab_size = in_matrix.shape[0]

        tie_weights = kwargs.get('tie_weights', True)
        if tie_weights:
            # Tie the model's weights
            gen_model.tie_weights()

        return gen_model

    def transfer(self, in_tokenizer, gen_tokenizer, gen_model, **kwargs):
        """
        Method that returns a new LM model with transferred embeddings.

        :param in_tokenizer: Any huggingface tokenizer
        :param gen_tokenizer: Any huggingface tokenizer
        :param gen_model: Any huggingface model
        :param kwargs: no kwargs

        :return: A new in_domain model
        """

        self.tokens_map = self.tokens_mapping(in_tokenizer, gen_tokenizer)
        in_matrix = self.embeddings_assignment(self.tokens_map, gen_model)
        in_model = self.update_model_embeddings(gen_model, in_matrix)

        return in_model


class FastVocabularyTransfer(VocabularyTransfer):

    def __init__(self):
        super().__init__()

    def tokens_mapping(self, in_tokenizer, gen_tokenizer, in_model=None, **kwargs):
        """
        This method establish a mapping between each token of
        the in-domain tokenizer (in_tokenizer) to one or more tokens from
        the general-purpose (gen_tokenizer) tokenizer.

        :param in_tokenizer: Any huggingface tokenizer
        :param gen_tokenizer: Any huggingface tokenizer
        :param in_model: A huggingface model corresponding to in_tokenizer
        :param kwargs: no kwargs

        :return: A dictionary, having size of the in_tokenizer vocabulary.
         Each key is the index corresponding to a token in the in-tokenizer.
         Values are lists of indexes to the tokens of gen_tokenizer.
        """

        gen_vocab = gen_tokenizer.get_vocab()
        in_vocab = in_tokenizer.get_vocab()
        ngram_vocab = in_tokenizer.ngram_vocab if hasattr(in_tokenizer, 'ngram_vocab') else {}

        tokens_map = {}
        for new_token, new_index in in_vocab.items():
            if new_token in gen_vocab:
                # if the same token exists in the old vocabulary, take its embedding
                old_index = gen_vocab[new_token]
                tokens_map[new_index] = [old_index]
            
            else:
                # if not, tokenize the new token using the old vocabulary
                new_token = re.sub('^(##|Ġ|▁)', '', new_token)
                if new_token in ngram_vocab:
                    token_partition = gen_tokenizer.tokenize(new_token.split('‗'), is_split_into_words=True)
                else:
                    token_partition = gen_tokenizer.tokenize(new_token)
                
                tokens_map[new_index] = [gen_vocab[old_token] for old_token in token_partition]

                # TODO: can we try to find a better way to aggregate the embeddings
                #  Calculate output embeddings

        return tokens_map

    def embeddings_assignment(self, tokens_map, gen_model, **kwargs):
        """
        Given a mapping between two tokenizers and a general-purpose model
        trained on gen_tokenizer, this method produces a new embedding matrix
        assigning embeddings from the gen_model.

        :param tokens_map: A mapping between new and old tokens. See tokens_mapping(...)
        :param gen_model: A huggingface model, e.g. bert
        :param kwargs: no kwargs

        :return: (2-d torch.Tensor) An embedding matrix with same size of tokens_map.
        """

        gen_matrix = gen_model.get_input_embeddings().weight
        in_matrix = torch.zeros(len(tokens_map), gen_matrix.shape[1])

        for new_index, old_indices in tokens_map.items():
            old_embedding = torch.mean(gen_matrix[old_indices], axis=0)
            in_matrix[new_index] = old_embedding

        return in_matrix
