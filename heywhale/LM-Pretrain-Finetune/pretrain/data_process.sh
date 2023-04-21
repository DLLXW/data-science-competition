python tools/preprocess_data.py \
       --input diag_train.json \
       --output-prefix desc_diag \
       --vocab vocab/vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type Huggingface \
       --split-sentences