"""BART model."""

import torch
from megatron import get_args, get_tokenizer
from megatron import mpu
from megatron.model.enums import AttnMaskType
from megatron.model.language_model import parallel_lm_logits
from megatron.model.language_model import get_language_model
from megatron.model import LayerNorm
from megatron.model.utils import openai_gelu, erf_gelu
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from .module import MegatronModule
from transformers import BertConfig, BertForMaskedLM

class BartModel(MegatronModule):
    def __init__(self):
        super().__init__()
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        config = BertConfig.from_pretrained('roberta-zh/base')
        tokenizer = get_tokenizer()
        config.vocab_size = tokenizer.vocab_size
        config.gradient_checkpointing = args.checkpoint_activations
        if args.num_layers is not None and args.num_decoder_layers is not None:
            encoder_layers = args.num_layers - args.num_decoder_layers
            config.encoder_layers = encoder_layers
            config.decoder_layers = args.num_decoder_layers
            config.num_hidden_layers = encoder_layers
        if args.hidden_size is not None:
            config.d_model = args.hidden_size
            config.encoder_ffn_dim = args.hidden_size * 4
            config.decoder_ffn_dim = args.hidden_size * 4
        if args.num_attention_heads is not None:
            config.encoder_attention_heads = args.num_attention_heads
            config.decoder_attention_heads = args.num_attention_heads
        self.language_model = BertForMaskedLM.from_pretrained('roberta-zh/base', config)
        self._language_model_key = 'language_model'

    def word_embeddings_weight(self):
        return self.language_model.get_input_embeddings().weight

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        assert input_tensor is None, "Huggingface Model Not Support Pipeline"

    def forward(self, input_ids, attn_mask, decoder_input, targets, use_decoder):
        lm_logits = self.language_model.model.encoder(
                            input_ids,
                            attention_mask=attn_mask)[0]
        
        if lm_logits.shape[:2] != targets.shape:
            print('lm_logits size:', lm_logits.size())
            import pdb; pdb.set_trace()

        if targets is None:
            return lm_logits, None
        else:
            if self.fp16_lm_cross_entropy:
                assert lm_logits.dtype == torch.half
                lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits, targets)
            else:
                lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits.float(),
                                                        targets)
            return lm_loss, None

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict()
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        # print(state_dict.keys())
        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
