from .tokenizer_transfer import (
    RandomInitializationTokenizerTransfer,
    OverlapTokenizerTransfer
)
from .clp_transfer import CLPTokenizerTransfer
from .fvt_transfer import FVTTokenizerTransfer
from .wechsel_transfer import WechselTokenizerTransfer
from .focus_transfer import FocusTokenizerTransfer
from .ramen_transfer import RamenTokenizerTransfer


def test_random_embedding_initialization():
    source_model_name = "FacebookAI/roberta-base"
    target_tokenizer_name = "phucdev/vi-spm-culturax-4g-sample"
    transfer_pipeline = RandomInitializationTokenizerTransfer(source_model_name, target_tokenizer_name)
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


def test_overlap_embedding_initialization():
    source_model_name = "FacebookAI/roberta-base"
    target_tokenizer_name = "phucdev/vi-spm-culturax-4g-sample"
    transfer_pipeline = OverlapTokenizerTransfer(source_model_name, target_tokenizer_name)
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


def test_ramen_embedding_initialization():
    source_model_name = "FacebookAI/roberta-base"
    target_model_name = "phucdev/vi-spm-culturax-4g-sample"
    transfer_pipeline = RamenTokenizerTransfer(
        source_model_name,
        target_model_name,
        aligned_data_path="data/parallel_data/OpenSubtitles",
        source_language_identifier="en",
        target_language_identifier="vi",
        corpus="OpenSubtitles"
    )
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


def test_wechsel_embedding_initialization():
    source_model_name = "FacebookAI/roberta-base"
    target_model_name = "phucdev/vi-spm-culturax-4g-sample"
    transfer_pipeline = WechselTokenizerTransfer(
        source_model_name,
        target_model_name,
        bilingual_dictionary_path="bilingual_dictionary/MUSE/en-vi.txt",
        source_language_identifier="en",
        target_language_identifier="vi",
    )
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


def test_focus_embedding_initialization():
    source_model_name = "FacebookAI/xlm-roberta-base"
    target_model_name = "phucdev/vi-spm-culturax-4g-sample"
    transfer_pipeline = FocusTokenizerTransfer(
        source_model_name,
        target_model_name,
        language_identifier="vi",
        target_training_data_path="data/culturax_vi/sample.jsonl",
        processes=1
    )
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


def test_fvt_embedding_initialization():
    source_model_name = "FacebookAI/xlm-roberta-base"
    target_model_name = "phucdev/vi-spm-culturax-4g-sample"
    transfer_pipeline = FVTTokenizerTransfer(
        source_model_name,
        target_model_name
    )
    target_model= transfer_pipeline.transfer()
    assert target_model is not None
