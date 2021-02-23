cd ..
python scripts/convert_bert_from_huggingface_to_uer.py --input_model_path /home/admins/pretrained/chinese_roberta_wwm_ext_pytorch/pytorch_model.bin \
                                                       --output_model_path models/chinese_roberta_wwm_ext_pytorch/pytorch_model_uer.bin \
                                                       --layers_num 12