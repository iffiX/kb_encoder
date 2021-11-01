import torch
import torch.nn as nn


class SharedVariable:
    def __init__(self, value=None):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


def modify_t5_attention_with_attr(t5_att, attr_weight):
    def forward(
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Copied from original T5 code, look for ### for modifications.

        Self-attention (if key_value_states is None) or attention
        over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or
        # (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert len(past_key_value) == 2, (
                f"past_key_value should have 2 past states: keys and values. "
                f"Got {len(past_key_value)} past states"
            )
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )

        def shape(states):
            """projection"""
            return states.view(
                batch_size, -1, t5_att.n_heads, t5_att.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return (
                states.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, t5_att.inner_dim)
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # t5_att-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # t5_att-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            t5_att.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            t5_att.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            t5_att.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states),
        # compatible with onnx op>9

        if position_bias is None:
            if not t5_att.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, t5_att.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if t5_att.gradient_checkpointing and t5_att.training:
                    position_bias.requires_grad = True
            else:
                position_bias = t5_att.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=t5_att.dropout, training=t5_att.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        ### Modification
        if attr_weight is not None:
            attn_output = unshape(
                torch.matmul(attn_weights * attr_weight.get(), value_states)
            )  # (batch_size, seq_length, dim)
        else:
            attn_output = unshape(
                torch.matmul(attn_weights, value_states)
            )  # (batch_size, seq_length, dim)

        attn_output = t5_att.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (t5_att.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

    t5_att.forward = forward


def modify_t5_model_with_attr(model):
    attr_weight = SharedVariable(1)
    att_layer = model.encoder.block[0].layer[0].SelfAttention
    modify_t5_attention_with_attr(att_layer, attr_weight)
    return attr_weight
