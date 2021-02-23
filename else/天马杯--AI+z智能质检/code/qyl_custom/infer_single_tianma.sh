cd ..
python inference/run_classifier_infer.py \
        --load_model_path ckpt_single/epoch_3_6766.bin \
        --vocab_path models/google_zh_vocab.txt \
        --config_path models/bert_base_config.json \
        --test_path datasets/tianma_cup/testB_group8_no_label.tsv \
        --seq_length 100 \
        --labels_num 2 \
        --encoder bert \
        --prediction_path prediction_single_B/subB_epoch3.csv \
        --output_prob
