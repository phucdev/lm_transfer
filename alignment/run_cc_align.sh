#!/bin/bash
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e
s=${1:-en}
t=${2:-vi}
src_tokenizer=${3:-''}
tgt_tokenizer=${4:-''}
echo "Example based on the ${s}->${t} alignment"

if [ ! -d data/ ]; then
  mkdir -p data;
fi

if [ ! -d res/ ]; then
  mkdir -p res;
fi

dico_train=data/${s}-${t}.0-5000.txt
if [ ! -f "${dico_train}" ]; then
  DICO=$(basename -- "${dico_train}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi

dico_test=data/${s}-${t}.5000-6500.txt
if [ ! -f "${dico_test}" ]; then
  DICO=$(basename -- "${dico_test}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi

src_emb=data/cc.${s}.300.vec
if [ ! -f "${src_emb}" ]; then
  EMB=$(basename -- "${src_emb}")
  EMB_GZ="${EMB}.gz"
  wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${EMB_GZ}" -P data/
  gunzip "data/${EMB_GZ}"
fi

tgt_emb=data/cc.${t}.300.vec
if [ ! -f "${tgt_emb}" ]; then
  EMB=$(basename -- "${tgt_emb}")
  EMB_GZ="${EMB}.gz"
  wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${EMB_GZ}" -P data/
  gunzip "data/${EMB_GZ}"
fi

output=res/cc.${s}-${t}.vec

python3 align.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
  --dico_train "${dico_train}" --dico_test "${dico_test}" --output "${output}" \
  --lr 25 --niter 10 --src_tokenizer "${src_tokenizer}" --tgt_tokenizer "${tgt_tokenizer}"
python3 eval.py --src_emb "${output}" --tgt_emb "${tgt_emb}" \
  --dico_test "${dico_test}"