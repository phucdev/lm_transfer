import pytest
from .tokenizer_transfer import (
    RandomInitializationTokenizerTransfer,
    OverlapTokenizerTransfer,
    CLPTokenizerTransfer
)


def test_random_embedding_initialization():
    source_model_name = "EleutherAI/pythia-410m"
    target_tokenizer_name = "malteos/gpt2-wechsel-german-ds-meg"
    transfer_pipeline = RandomInitializationTokenizerTransfer(source_model_name, target_tokenizer_name)
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


def test_overlap_embedding_initialization():
    source_model_name = "EleutherAI/pythia-410m"
    target_tokenizer_name = "malteos/gpt2-wechsel-german-ds-meg"
    transfer_pipeline = OverlapTokenizerTransfer(source_model_name, target_tokenizer_name)
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


def test_clp_embedding_initialization():
    source_model_name = "EleutherAI/pythia-410m"
    target_model_name = "malteos/gpt2-wechsel-german-ds-meg"
    transfer_pipeline = CLPTokenizerTransfer(
        source_model_name,
        target_model_name,
        helper_model_name_or_path=target_model_name
    )
    target_model = transfer_pipeline.transfer()
    assert target_model is not None
