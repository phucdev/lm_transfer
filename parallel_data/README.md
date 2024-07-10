# Leverage parallel data like RAMEN

OpenSubtitles Sample from RAMEN paper

```bash
./fast_align/build/fast_align -i parallel_data/sample-parallel-en-vi.txt -d -o -v -I 10 > parallel_data/forward.en-vi
```

```bash
./fast_align/build/fast_align -i parallel_data/sample-parallel-en-vi.txt -d -o -v -r -I 10 > parallel_data/reverse.en-vi
```

Full OpenSubtitles
```bash
./fast_align/build/fast_align -i parallel_data/full/parallel-en-vi.txt -d -o -v -I 10 > parallel_data/full/forward.en-vi
```

```bash
./fast_align/build/fast_align -i parallel_data/full/parallel-en-vi.txt -d -o -v -r -I 10 > parallel_data/full/reverse.en-vi
```

Symmetric Alignment
```bash
./fast_align/build/atools -i parallel_data/full/forward.en-vi -j parallel_data/full/reverse.en-vi -c grow-diag-final-and > parallel_data/full/align.en-vi
```
Alternatively use `awesome-align` to get alignments

Get probs
```bash
python src/get_prob_para.py --bitxt parallel_data/sample-parallel-en-vi.txt --align parallel_data/awesome_aligned.txt --save parallel_data/probs.para.en-vi.pth
```

