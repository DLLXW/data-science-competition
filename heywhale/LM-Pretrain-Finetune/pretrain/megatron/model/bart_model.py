"""BART model."""

import torch
from transformers.utils.dummy_pt_objects import BertModel

from megatron import get_args, get_tokenizer
from megatron import mpu
from megatron.model.enums import AttnMaskType
from megatron.model.language_model import parallel_lm_logits
from megatron.model.language_model import get_language_model
from megatron.model import LayerNorm
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from .module import MegatronModule
from transformers import BartForConditionalGeneration as HFBartModel
# from megatron.model.modeling_bart import BartForConditionalGeneration as HFBartModel
from transformers import BertConfig, BartConfig

class BartModel(MegatronModule):
    def __init__(self):
        super().__init__()
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        config = BartConfig.from_pretrained(args.vocab_file)  # vocab file path also contains config.json
        # encoder_config = BertConfig.from_pretrained(model_path)
        tokenizer = get_tokenizer()
        config.vocab_size = tokenizer.vocab_size
        config.gradient_checkpointing = args.checkpoint_activations
        # if args.num_layers is not None and args.num_decoder_layers is not None:
        #     encoder_layers = args.num_layers - args.num_decoder_layers
        #     config.encoder_layers = encoder_layers
        #     config.decoder_layers = args.num_decoder_layers
        #     config.num_hidden_layers = encoder_layers
        # if args.hidden_size is not None:
        #     config.d_model = args.hidden_size
        #     config.encoder_ffn_dim = args.hidden_size * 4
        #     config.decoder_ffn_dim = args.hidden_size * 4
        # if args.num_attention_heads is not None:
        #     config.encoder_attention_heads = args.num_attention_heads
        #     config.decoder_attention_heads = args.num_attention_heads
        self.language_model = HFBartModel(config)
        self._language_model_key = 'language_model'
        ckpt = torch.load("/home/trojanjet/project/weiqin/diag/bart/bart-chinese/pytorch_model.bin")
        ckpt.pop("final_logits_bias")
        ckpt.pop("model.shared.weight")
        ckpt.pop("model.encoder.embed_tokens.weight")
        ckpt.pop("model.decoder.embed_tokens.weight")
        ckpt.pop("lm_head.weight")
        self.language_model.load_state_dict(ckpt,strict=False)
        print('pass to load pretrain model')
        
        self.num_decoder_layers = args.num_decoder_layers
        print("weiqin debug config..........")
        config.save_pretrained("./checkpoints") #

    def word_embeddings_weight(self):
        return self.language_model.get_input_embeddings().weight

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        assert input_tensor is None, "Huggingface Model Not Support Pipeline"

    def forward(self, input_ids, attn_mask, decoder_input, targets, use_decoder):
        outputs = self.language_model(
            input_ids, attention_mask=attn_mask,
            decoder_input_ids=decoder_input
        )
        lm_logits = outputs[0]

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
