def get_bert_special_tokens():
    return {
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]"
    }


def get_deberta_special_tokens():
    return {
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "bos_token": "[CLS]",
        "eos_token": "[SEP]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]"
    }


def roberta_special_tokens():
    return {
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "cls_token": "<s>",
        "sep_token": "</s>",
        "mask_token": "<mask>"
    }


def xlm_roberta_special_tokens():
    return {
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "cls_token": "<s>",
        "sep_token": "</s>",
        "mask_token": "<mask>"
    }
