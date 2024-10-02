import os
from lm_transfer.embedding_initialization.tokenizer_transfer import (
    RandomInitializationTokenizerTransfer,
    OverlapTokenizerTransfer
)
# from lm_transfer.embedding_initialization.clp_transfer import CLPTokenizerTransfer
from lm_transfer.embedding_initialization.fvt_transfer import FVTTokenizerTransfer
from lm_transfer.embedding_initialization.wechsel_transfer import WechselTokenizerTransfer
from lm_transfer.embedding_initialization.focus_transfer import FocusTokenizerTransfer
from lm_transfer.embedding_initialization.ramen_transfer import RamenTokenizerTransfer


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def test_random_embedding_initialization():
    source_model_name = "FacebookAI/roberta-base"
    target_tokenizer_name = "phucdev/vi-bpe-culturax-4g-sample"
    transfer_pipeline = RandomInitializationTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        target_model_path=os.path.join(parent_dir, "models/test/random_initialization")
    )
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


def test_overlap_embedding_initialization():
    source_model_name = "FacebookAI/roberta-base"
    target_tokenizer_name = "phucdev/vi-bpe-culturax-4g-sample"
    transfer_pipeline = OverlapTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        target_model_path=os.path.join(parent_dir, "models/test/overlap_initialization")
    )
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


def test_ramen_embedding_initialization():
    source_model_name = "FacebookAI/roberta-base"
    target_model_name = "phucdev/vi-bpe-culturax-4g-sample"
    transfer_pipeline = RamenTokenizerTransfer(
        source_model_name,
        target_model_name,
        aligned_data_path=os.path.join(parent_dir, "data/parallel_data/OpenSubtitles"),
        source_language_identifier="en",
        target_language_identifier="vi",
        corpus="OpenSubtitles",
        target_model_path=os.path.join(parent_dir, "models/test/ramen_initialization")
    )
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


def test_wechsel_embedding_initialization():
    source_model_name = "FacebookAI/roberta-base"
    target_model_name = "phucdev/vi-bpe-culturax-4g-sample"
    transfer_pipeline = WechselTokenizerTransfer(
        source_model_name,
        target_model_name,
        bilingual_dictionary=os.path.join(parent_dir, "bilingual_dictionary/MUSE/en-vi.txt"),
        source_language_identifier="en",
        target_language_identifier="vi",
        target_model_path=os.path.join(parent_dir, "models/test/wechsel_initialization")
    )
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


def test_wechsel_aligned_embedding_initialization():
    source_model_name = "FacebookAI/roberta-base"
    target_model_name = "phucdev/vi-bpe-culturax-4g-sample"
    transfer_pipeline = WechselTokenizerTransfer(
        source_model_name,
        target_model_name,
        align_strategy=None,
        use_subword_info=False,
        bilingual_dictionary=None,
        source_language_identifier="en",
        target_language_identifier="vi",
        target_model_path=os.path.join(parent_dir, "models/test/wechsel_aligned_initialization"),
    )
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


def test_wechsel_overlap_embedding_initialization():
    source_model_name = "FacebookAI/roberta-base"
    target_model_name = "phucdev/vi-bpe-culturax-4g-sample"
    transfer_pipeline = WechselTokenizerTransfer(
        source_model_name,
        target_model_name,
        bilingual_dictionary=os.path.join(parent_dir, "bilingual_dictionary/MUSE/en-vi.txt"),
        source_language_identifier="en",
        target_language_identifier="vi",
        target_model_path=os.path.join(parent_dir, "models/test/wechsel_overlap_initialization"),
        leverage_overlap=True,
        overwrite_with_overlap=True
    )
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


# def test_clp_embedding_initialization():
#     source_model_name = "EleutherAI/pythia-410m"
#     target_model_name = "malteos/gpt2-wechsel-german-ds-meg"
#     transfer_pipeline = CLPTokenizerTransfer(
#         source_model_name,
#         target_model_name,
#         helper_model_name_or_path=target_model_name,
#         target_model_path="models/test/clp_initialization"
#     )
#     target_model = transfer_pipeline.transfer()
#     assert target_model is not None


def test_focus_monolingual_embedding_initialization():
    source_model_name = "FacebookAI/roberta-base"
    target_model_name = "phucdev/vi-bpe-culturax-4g-sample"
    transfer_pipeline = FocusTokenizerTransfer(
        source_model_name,
        target_model_name,
        language_identifier="vi",
        target_training_data_path=os.path.join(parent_dir, "data/culturax_vi/sample.jsonl"),
        processes=1,
        target_model_path=os.path.join(parent_dir, "models/test/focus_monolingual_initialization")
    )
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


def test_overlap_multilingual_embedding_initialization():
    source_model_name = "FacebookAI/xlm-roberta-base"
    target_tokenizer_name = "phucdev/vi-spm-culturax-4g-sample"
    transfer_pipeline = OverlapTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        target_model_path=os.path.join(parent_dir, "models/test/overlap_multilingual_initialization")
    )
    target_model = transfer_pipeline.transfer()
    assert target_model is not None


def test_fvt_embedding_initialization():
    source_model_name = "FacebookAI/xlm-roberta-base"
    target_model_name = "phucdev/vi-spm-culturax-4g-sample"
    transfer_pipeline = FVTTokenizerTransfer(
        source_model_name,
        target_model_name,
        target_model_path=os.path.join(parent_dir, "models/test/fvt_initialization")
    )
    target_model= transfer_pipeline.transfer()
    assert target_model is not None


def test_focus_multilingual_embedding_initialization():
    source_model_name = "FacebookAI/xlm-roberta-base"
    target_model_name = "phucdev/vi-spm-culturax-4g-sample"
    transfer_pipeline = FocusTokenizerTransfer(
        source_model_name,
        target_model_name,
        language_identifier="vi",
        target_training_data_path=os.path.join(parent_dir, "data/culturax_vi/sample.jsonl"),
        processes=1,
        target_model_path=os.path.join(parent_dir, "models/test/focus_multilingual_initialization")
    )
    target_model = transfer_pipeline.transfer()
    assert target_model is not None
