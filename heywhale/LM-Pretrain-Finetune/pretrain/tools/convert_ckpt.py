import torch
import os, sys
from torch.serialization import save
# from transformers import BartForConditionalGeneration, BartConfig

def main():
    ckpt_dir = sys.argv[1]
    save_dir = sys.argv[2]

    ckpt = ckpt_dir + '/model_optim_rng.pt'
    states = torch.load(ckpt, map_location='cpu')
    new_states = {}
    vocab_size = 0
    for key, val in states['model']['language_model'].items():
        new_states[key.replace('model.', '', 1)] = val.float()
        if '.weight' in key:
            vocab_size = max(vocab_size, val.size(0))
    new_states.pop('final_logits_bias', None)
    new_states.pop('lm_head.weight', None)
    save_path = save_dir + '/pytorch_model.bin'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(new_states, save_path)
    print(ckpt + ' -> ' + save_path)
    # print('using config {}, vocab_size: {}'.format(config.name_or_path, config.vocab_size))
    print('model vocab_size: {}'.format(vocab_size))

if __name__ == '__main__':
    main()
