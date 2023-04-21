"""BART model."""

import torch
from transformers.utils.dummy_pt_objects import BertModel

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
from megatron.model.modeling_cpt import CPTForConditionalGeneration as HFBartModel
from transformers import BertConfig, BartConfig

class CPTModel(MegatronModule):
    def __init__(self):
        super().__init__()
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        # config = BartConfig.from_pretrained(args.vocab_file)  # vocab file path also contains config.json
        model_path = 'roberta_zh'
        config = BartConfig.from_pretrained(args.vocab_file)  # vocab file path also contains config.json
        # encoder_config = BertConfig.from_pretrained(model_path)
        tokenizer = get_tokenizer()
        config.vocab_size = tokenizer.vocab_size
        config.gradient_checkpointing = args.checkpoint_activations
        # if args.num_layers is not None and args.num_decoder_layers is not None:
        #     # encoder_layers = args.num_layers - args.num_decoder_layers
        #     encoder_layers = args.num_layers
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
        # self.language_model = HFBartModel(config, encoder_config)
        self.language_model = HFBartModel(config)
        encoder_state = torch.load('/home/trojanjet/project/weiqin/diag/bart/cpt_large/pytorch_model.bin', map_location='cpu')
        encoder_state.pop('model.encoder.embeddings.position_ids')
        encoder_state.pop('model.encoder.embeddings.position_embeddings.weight')
        self.language_model.load_state_dict(encoder_state,strict=False)
    
        self._language_model_key = 'language_model'
        self.num_decoder_layers = args.num_decoder_layers
        config.save_pretrained("./checkpoints/cpt-large") #
        

    def word_embeddings_weight(self):
        return self.language_model.get_input_embeddings().weight

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        assert input_tensor is None, "Huggingface Model Not Support Pipeline"

    def forward(self, input_ids, attn_mask, decoder_input, targets, use_decoder):
        batch_ids = torch.arange(input_ids.size(0)).to(use_decoder)
        use_decoder_batch_ids = batch_ids[use_decoder == 1]
        no_use_decoder_batch_ids = batch_ids[use_decoder != 1]
        reorder_batch_ids = torch.cat([use_decoder_batch_ids, no_use_decoder_batch_ids], dim=0)
        input_ids = input_ids[reorder_batch_ids]
        attn_mask = attn_mask[reorder_batch_ids]
        decoder_input = decoder_input[reorder_batch_ids]
        num_use_decoder = use_decoder_batch_ids.size(0)

        token_type_ids = torch.ones_like(input_ids)
        token_type_ids[:num_use_decoder, :] = 1

        encoder_outputs = self.language_model.model.encoder(
                            input_ids,
                            attention_mask=attn_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True,
                            return_dict=True)

        encoder_outputs_for_decoder = encoder_outputs.hidden_states[-self.num_decoder_layers-1]
        encoder_output = encoder_outputs.last_hidden_state
        
        decoder_lm_logits = None
        if num_use_decoder > 0:
            decoder_lm_logits = self.language_model(
                input_ids[:num_use_decoder], attention_mask=attn_mask[:num_use_decoder],
                decoder_input_ids=decoder_input[:num_use_decoder],
                encoder_outputs=encoder_outputs_for_decoder[:num_use_decoder])[0]

        encoder_lm_logits = None
        if num_use_decoder < input_ids.size(0):
            encoder_lm_logits = self.language_model.lm_head(encoder_output[num_use_decoder:]) \
                    + self.language_model.final_logits_bias

        if decoder_lm_logits is None:
            reorder_lm_logits = encoder_lm_logits
        elif encoder_lm_logits is None:
            reorder_lm_logits = decoder_lm_logits
        else:
            reorder_lm_logits = torch.cat([decoder_lm_logits, encoder_lm_logits], dim=0)
        _, reverse_batch_ids = reorder_batch_ids.sort(dim=0)
        lm_logits = reorder_lm_logits[reverse_batch_ids]
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
