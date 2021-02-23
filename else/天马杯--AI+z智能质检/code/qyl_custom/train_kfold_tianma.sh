cd ..
python run_classifier_cv.py --pretrained_model_path models/chinese_roberta_wwm_ext_pytorch/pytorch_model_uer.bin \
                            --vocab_path models/google_zh_vocab.txt --output_model_path ckpt/model-0.bin \
                            --config_path models/bert_base_config.json \
                            --train_path datasets/tianma_cup/train_group8.tsv --train_features_path datasets/tianma_cup/features/train/model-0.npy \
                            --folds_num 5 --epochs_num 4 --batch_size 64 --encoder bert --seq_length 100