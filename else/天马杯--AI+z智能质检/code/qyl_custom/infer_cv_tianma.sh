cd ..
python3 inference/run_classifier_infer_cv.py --load_model_path ckpt/model-0.bin \
                                             --vocab_path models/google_zh_vocab.txt \
                                             --config_path models/bert_base_config.json \
                                             --test_path datasets/tianma_cup/testB_group8_no_label.tsv --test_features_path datasets/tianma_cup/features/testB/model-0.npy \
                                             --folds_num 5 --labels_num 2 --encoder bert --seq_length 100