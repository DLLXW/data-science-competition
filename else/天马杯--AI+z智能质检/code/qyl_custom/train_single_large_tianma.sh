cd ..
python run_classifier.py --pretrained_model_path models/chinese_roberta_wwm_large_ext_pytorch/pytorch_model_uer.bin \
                            --vocab_path models/google_zh_vocab.txt --output_model_path ckpt_single_large/model_tianma.bin \
                            --config_path models/bert_large_config.json \
                            --train_path datasets/tianma_cup/train_group8_demo.tsv \
                            --dev_path datasets/tianma_cup/dev_group8.tsv \
                            --epochs_num 5 --batch_size 4 --encoder bert --seq_length 100