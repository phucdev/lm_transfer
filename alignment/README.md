## Alignment of Word Embeddings

This is code from: https://github.com/facebookresearch/fastText/tree/main/alignment

This directory provides code for learning alignments between word embeddings in different languages.

The code is in Python 3 and requires [NumPy](http://www.numpy.org/).

The script `example.sh` shows how to use this code to learn and evaluate a bilingual alignment of word embeddings.
In this adapted version, we align the fastText embeddings that were trained on the Common Crawl.

For our language pair (English->Vietnamese) run the script like this:
```bash
./example.sh en vi FacebookAI/roberta-base phucdev/vi-bpe-culturax-4g-sample
```

The word embeddings used in [1] can be found on the [fastText project page](https://fasttext.cc) and the supervised bilingual lexicons on the [MUSE project page](https://github.com/facebookresearch/MUSE).

### Supervised alignment

The script `align.py` aligns word embeddings from two languages using a bilingual lexicon as supervision.
The details of this approach can be found in [1].

### Download

Wikipedia fastText embeddings aligned with our method can be found [here](https://fasttext.cc/docs/en/aligned-vectors.html).

### References

If you use the supervised alignment method, please cite:

[1] A. Joulin, P. Bojanowski, T. Mikolov, H. Jegou, E. Grave, [*Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion*](https://arxiv.org/abs/1804.07745)

```
@InProceedings{joulin2018loss,
    title={Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion},
    author={Joulin, Armand and Bojanowski, Piotr and Mikolov, Tomas and J\'egou, Herv\'e and Grave, Edouard},
    year={2018},
    booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
}
```
