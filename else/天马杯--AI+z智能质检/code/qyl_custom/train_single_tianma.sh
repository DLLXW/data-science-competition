cd ..
python run_classifier.py --pretrained_model_path models/chinese_roberta_wwm_ext_pytorch/pytorch_model_uer.bin \
                            --vocab_path models/google_zh_vocab.txt --output_model_path ckpt_single/model_tianma.bin \
                            --config_path models/bert_base_config.json \
                            --train_path datasets/tianma_cup/train_dev_group8.tsv \
                            --dev_path datasets/tianma_cup/dev_group8.tsv \
                            --epochs_num 8 --batch_size 64 --encoder bert --seq_length 100