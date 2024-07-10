# MA-Exploration
Repo for exploring topics for a master thesis in cross-lingual transfer

The script: src/mf_initialization.py
transfers word embeddings from a source model to the target language by using the CLP-Transfer method, but combines it
with matrix factorization. The main idea is to decompose the original word embeddings matrix into two matrices, 
a lower dimensional word embedding matrix and an up-projection matrix that encodes general linguistic information.
The new lower dimensional word embedding matrix is initialized with the CLP-Transfer method and is then up-projected
to the original dimensionality. The up-projection matrix is reused, which means that we leverage information from the
entire source language word embedding matrix, not just from the overlapping words.

The script: src/initialization_with_bilingual_dictionary_frequency.py
We walk through the dictionary and tokenize the source language word with the source language tokenizer and the target 
language word with the target language tokenizer. We track how often a source language word occurs with a target 
language word in translation pairs. We use this normalized frequency matrix to determine how much a source language
word embeddings contributes to the initialization of a target language word embedding. 
