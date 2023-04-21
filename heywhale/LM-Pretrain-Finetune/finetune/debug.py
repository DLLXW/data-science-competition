# from transformers import BartForConditionalGeneration, BartTokenizer
# import torch
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")#fnlp/bart-base-chinese
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
# for name,param in model.named_parameters():
#     print(name)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# articles = ['This is an example.']  # put your articles here
# dct = tokenizer.batch_encode_plus(articles, max_length=1024, return_tensors="pt", pad_to_max_length=True)  # you can change max_length if you want
# model.to(device)
# summaries = model.generate(
#     input_ids=dct["input_ids"].to(device),
#     attention_mask=dct["attention_mask"].to(device),
#     num_beams=4,
#     length_penalty=2.0,
#     max_length=142,  # +2 from original because we start at step=1 and stop before max_length
#     min_length=56,  # +1 from original because we start at step=1
#     no_repeat_ngram_size=3,
#     early_stopping=True,
#     do_sample=False,
# )  # change these arguments if you want

# dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
# print(dec)
# stories=['2 4 5 146 18 19','3 21']
# desc=tokenizer.batch_encode_plus(stories, max_length=8, return_tensors="pt", pad_to_max_length=True)
# print(desc)
# weight=torch.load('../CPT-master/pretrain/checkpoints/bart-large/iter_0002000/mp_rank_00/model_optim_rng.pt')   #预训练路径

# weight=torch.load('/home/trojanjet/project/weiqin/diag/CPT-master/pretrain/checkpoints/bart-large/iter_0004000/mp_rank_00/model_optim_rng.pt')   #预训练路径
# torch.save(weight['model']['language_model'],'./custom_pretrain/pytorch_model.bin')