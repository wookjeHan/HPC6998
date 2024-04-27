import huggingface.transformers.models.gpt_neo.modeling_gpt_neo as gpt_neo
from modeling.multi_stream_attention import NgramMultiheadAttention




def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
            need_weights=True, static_kv=False,
            self_attn_mask=None,
            ngram_mask_matrix=None,
            i_buckets_main_stream=None,
            i_bucket_relative_stream=None,
            real_positions=None
            ):
    pass


class GPTNeoMultiStreamAttention(GPTNeoSelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._attn = NgramMultiheadAttention()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        self._attn(
            quer=query,
            key=key,
            value=value,
            self_attn_mask=attention_mask,
            ngram_mask_matrix=head_mask)

