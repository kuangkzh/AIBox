import math
import warnings

import torch
from torch import nn
from torch.nn import LayerNorm
# from transformers.models.bloom.modeling_bloom import BloomBlock, BloomModel, BloomForCausalLM
from transformers.models.bloom.modeling_bloom import (
    BloomBlock, BloomModel, BloomForCausalLM, BloomConfig, \
    BloomAttention, BloomGelu, BloomMLP, Optional, Tuple,
    BloomPreTrainedModel, PreTrainedModel, BaseModelOutputWithPastAndCrossAttentions, \
    logger, Union,
    CausalLMOutputWithCrossAttentions,
    CrossEntropyLoss,
    CausalLMOutputWithCrossAttentions,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


tokenizer = AutoTokenizer.from_pretrained("/home/nlper_data/kuangzh/models/YeungNLP/firefly-1b4")
bloom_model = AutoModelForCausalLM.from_pretrained("/home/nlper_data/kuangzh/models/YeungNLP/firefly-1b4", device_map="auto", torch_dtype=torch.float16).to("cuda")
config = bloom_model.config


class MainModelAttention(nn.Module):
    # inherit from transformers.models.bert.modeling_bert.BertSelfAttention
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        cross_hidden_states=None,
        cross_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        cross_hidden_states = hidden_states if cross_hidden_states is None else cross_hidden_states
        cross_attention_mask = attention_mask if cross_attention_mask is None else cross_attention_mask

        query_layer = self.query(hidden_states)
        key_layer = self.key(cross_hidden_states)
        value_layer = self.value(cross_hidden_states)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

# class MainModelBlock(BloomBlock):
#     def __init__(self, config):
#         super().__init__(config)


# class MainModel(BloomModel):
#     def __init__(self, config):
#         super().__init__(config)
#
#         self.embed_dim = config.hidden_size
#         self.num_heads = config.n_head
#
#         # Embedding + LN Embedding
#         self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
#         self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
#
#         # Transformer blocks
#         self.h = nn.ModuleList([BloomBlock(config) for _ in range(config.num_hidden_layers)])
#
#         # Final Layer Norm
#         self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
#
#         self.gradient_checkpointing = False
#
#         # Initialize weights and apply final processing
#         self.post_init()



class MainModelBlock(BloomBlock):
    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.lora_input = nn.Linear(config.hidden_size, config.hidden_size // 8)
        self.lora_output = nn.Linear(config.hidden_size // 8, config.hidden_size)
        self.lora_gelu = BloomGelu()
        # self.cross_mlp = BloomMLP(config)
        self.cross_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.cross_attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)

        # 初始化权重
        # nn.init.normal_(self.lora_input.weight, mean=0, std=0)
        # nn.init.normal_(self.lora_output.weight, mean=0, std=0)

    def forward(
            self,
            hidden_states: torch.Tensor,
            cross_hidden_states: torch.Tensor,
            alibi: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            head_mask: Optional[torch.Tensor] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        lora_hidden = self.lora_input(hidden_states)
        lora_res = self.lora_output(self.lora_gelu(lora_hidden))

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm: # cross_attention 没有加输入的layer norm residual
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # 添加cross attention的调用
        cross_attention_output, _ = self.cross_attention(hidden_states, cross_hidden_states, cross_hidden_states)
        cross_attention_layernorm_output = self.cross_layernorm(cross_attention_output)


        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
            # cross_residual = cross_attention_layernorm_output
        else:
            residual = attention_output
            # cross_residual = cross_attention_output

        layernorm_output += cross_attention_layernorm_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        # # mlp
        # cross_mlp_output = self.cross_mlp(cross_attention_layernorm_output, cross_residual)
        output += lora_res



        #
        # # 然后可以根据需要处理cross_attention_output，例如将其添加到output
        # hidden_states = hidden_states + cross_attention_output
        # output += cross_attention_output


        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class MainModel(BloomModel):
    def __init__(self, config: BloomConfig):
        super().__init__(config)

        # Transformer blocks
        # self.h = nn.ModuleList([BloomBlock(config) for _ in range(config.num_hidden_layers)])
        self.h = nn.ModuleList([MainModelBlock(config) for _ in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def load_from_bloom(self, bloom_model, init_std=0.0):
        match_res = self.load_state_dict(bloom_model.state_dict(), strict=False)
        warnings.warn("Load Result:")
        print(match_res)
        # 初始化lora层
        for block in self.h:
            nn.init.normal_(block.lora_input.weight, mean=0, std=init_std)
            nn.init.normal_(block.lora_output.weight, mean=0, std=init_std)
            # 初始化偏置
            nn.init.zeros_(block.lora_input.bias)
            nn.init.zeros_(block.lora_output.bias)

            # nn.init.xavier_uniform_(block.cross_attention.in_proj_weight)
            # nn.init.xavier_uniform_(block.cross_attention.out_proj.weight)
            nn.init.normal_(block.cross_attention.in_proj_weight, mean=0, std=init_std)
            nn.init.normal_(block.cross_attention.out_proj.weight, mean=0, std=init_std)
            nn.init.constant_(block.cross_attention.in_proj_bias, 0)
            nn.init.constant_(block.cross_attention.out_proj.bias, 0)

            # nn.init.normal_(block.cross_mlp.dense_h_to_4h.weight, mean=0, std=init_std)
            # nn.init.normal_(block.cross_mlp.dense_4h_to_h.weight, mean=0, std=init_std)
            # nn.init.zeros_(block.cross_mlp.dense_h_to_4h.bias)
            # nn.init.zeros_(block.cross_mlp.dense_4h_to_h.bias)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            cross_input: Optional[torch.Tensor] = None,  # 添加cross attention的输入
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    cross_input,  # 将cross attention的输入传入
                    alibi,
                    causal_mask,
                    layer_past,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    cross_hidden_states=cross_input,  # 将cross attention的输入传入
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class MainModelForCausalLM(BloomForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = MainModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def load_from_bloom(self, bloom_causal_lm_model, init_std=0.0):
        self.transformer.load_from_bloom(bloom_causal_lm_model.transformer, init_std)
        self.lm_head.load_state_dict(bloom_causal_lm_model.lm_head.state_dict(), strict=False)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            cross_input: Optional[torch.Tensor] = None,  # 添加cross attention的输入
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if cross_input is None and hasattr(self, "_cross_input"):
            cross_input = self._cross_input

        transformer_outputs = self.transformer(
            input_ids,
            cross_input,  # 添加cross attention的输入
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )