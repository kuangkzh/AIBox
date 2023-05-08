import math

import torch
from torch import nn
from torch.nn import LayerNorm
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomModel, BloomForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("/home/nlper_data/kuangzh/models/YeungNLP/firefly-1b4")
model = AutoModelForCausalLM.from_pretrained("/home/nlper_data/kuangzh/models/YeungNLP/firefly-1b4", device_map="auto", torch_dtype=torch.float16).to("cuda")


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
        past_key_value=None,
        output_attentions=False,
    ):

        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

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

        context_layer = self.dense(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class MainModelCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 定义cross attention层
        self.cross_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.cross_key = nn.Linear(config.hidden_size, config.hidden_size)
        self.cross_value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
            self,
            hidden_states,
            cross_hidden_states=None,
            cross_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):

        # cross attention层
        mixed_query_layer = self.cross_query(hidden_states)
        cross_key_layer = self.cross_key(cross_hidden_states)
        cross_value_layer = self.cross_value(cross_hidden_states)
        cross_attention_scores = torch.matmul(mixed_query_layer, cross_key_layer.transpose(-1, -2))  # context_layer ?
        cross_attention_scores = cross_attention_scores / math.sqrt(self.attention_head_size)
        if cross_attention_mask is not None:
            cross_attention_scores = cross_attention_scores + cross_attention_mask
        cross_attention_probs = nn.functional.softmax(cross_attention_scores, dim=-1)
        cross_attention_probs = self.dropout(cross_attention_probs)
        cross_context_layer = torch.matmul(cross_attention_probs, cross_value_layer)

        # 残差连接
        context_layer = hidden_states + cross_context_layer

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = self.dense(context_layer)

        outputs = (context_layer, cross_attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# class MainModelBlock(BloomBlock):
#     def __init__(self, config):
#         super().__init__(config)

class MainModelBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MainModelAttention(config)
        self.cross_attention = MainModelCrossAttention(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            cross_hidden_states=None,
            cross_attention_mask=None,
            output_attentions=False,
    ):
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions = output_attentions,
        )
        attention_output = attention_outputs[0]
        outputs = attention_outputs[1:]


        cross_attention_outputs = self.cross_attention(
            hidden_states,
            cross_hidden_states,
            cross_attention_mask,
            output_attentions = output_attentions,
        )
        cross_attention_output = cross_attention_outputs[0]
        cross_outputs = cross_attention_outputs[1:]

        return outputs

class MainModel(BloomModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Transformer blocks
        # self.h = nn.ModuleList([BloomBlock(config) for _ in range(config.num_hidden_layers)])
        self.h = nn.ModuleList([MainModelBlock(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
