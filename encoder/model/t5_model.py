import torch as t
import warnings
from transformers.models.t5.modeling_t5 import *
from fairscale.nn import Pipe
from fairscale.nn.pipe import checkpoint
from fairscale.nn.pipe.checkpoint import (
    save_rng_states,
    restore_rng_states,
    enable_checkpointing,
    enable_recomputing,
)


def none_tensor(batch_size):
    return t.tensor([[]] * batch_size)


def is_none_tensor(tensor):
    return tensor.numel() == 0


def move_or_use_existing_tensor(cache, key, tensor, device):
    if key in cache:
        return cache[key]
    else:
        cache[key] = tensor.to(device)
        return cache[key]


def checkpoint_forward(
    ctx, phony, recomputed, rng_states, function, input_atomic, *input,
):
    ctx.recomputed = recomputed
    ctx.rng_states = rng_states

    save_rng_states(input[0].device, ctx.rng_states)

    ctx.function = function
    ctx.input_atomic = input_atomic

    # Save tensors needed for backward to prevent memory leaks.
    # Saved tensors are not "used" but "kept".
    require_backward_input = [i for i in input if i.requires_grad]
    ctx.save_for_backward(*require_backward_input)

    with torch.no_grad(), enable_checkpointing():
        output = function(input[0] if input_atomic else input)

    return output


def checkpoint_backward(
    ctx, *grad_output,
):
    # Get output needed for the backward pass from the queue
    # shared by checkpoint and recompute stage
    output, input_leaf = ctx.recomputed.pop()

    if isinstance(output, tuple):
        tensors = output
    else:
        tensors = (output,)
    if any(y.requires_grad for y in tensors):
        require_backward_tensors, require_backward_grads = [], []
        for x, g in zip(tensors, grad_output):
            if x.requires_grad:
                require_backward_tensors.append(x)
                require_backward_grads.append(g)
        torch.autograd.backward(require_backward_tensors, require_backward_grads)

    grad_input = [None, None, None, None, None]
    grad_input.extend(x.grad for x in input_leaf)
    return tuple(grad_input)


def recompute_forward(
    ctx, phony, recomputed, rng_states, function, input_atomic, *input,
):
    ctx.recomputed = recomputed
    ctx.rng_states = rng_states

    ctx.function = function
    ctx.input_atomic = input_atomic

    ctx.save_for_backward(*input)
    return phony


def recompute_backward(ctx, *grad_output):
    input = ctx.saved_tensors
    input_leaf = tuple(x.detach().requires_grad_(x.requires_grad) for x in input)

    with restore_rng_states(input[0].device, ctx.rng_states):
        with torch.enable_grad(), enable_recomputing():
            output = ctx.function(input_leaf[0] if ctx.input_atomic else input_leaf)
    ctx.recomputed.append((output, input_leaf))

    grad_input = [None, None, None, None, None]
    grad_input.extend(None for _ in ctx.saved_tensors)
    return tuple(grad_input)


checkpoint.Checkpoint.forward = checkpoint_forward
checkpoint.Checkpoint.backward = checkpoint_backward
checkpoint.Recompute.forward = recompute_forward
checkpoint.Recompute.backward = recompute_backward


class T5BlockForPipeline(T5Block):
    def __init__(self, config, index, has_relative_attention_bias=False):
        super(T5BlockForPipeline, self).__init__(config, has_relative_attention_bias)
        self.index = index
        self.device = None
        self.forward_flags = {
            "use_cache": False,
            "output_attentions": False,
            "return_dict": True,
        }
        self.forward_inputs = {
            "attention_mask": None,
            "encoder_attention_mask": None,
            "head_mask": None,
            "cross_attn_head_mask": None,
            "past_key_values": None,
        }
        self.forward_results = {
            "present_key_value_state": [],
            "hidden_states": [],
            "attentions": [],
            "cross_attentions": [],
        }

    def set_forward_flags(
        self, use_cache=None, output_attentions=None, return_dict=None,
    ):
        self.forward_flags["use_cache"] = use_cache
        self.forward_flags["output_attentions"] = output_attentions
        self.forward_flags["return_dict"] = return_dict

    def set_forward_inputs(
        self,
        attention_mask=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
    ):
        # Note: since deque length in the checkpoint is 1,
        # therefore forwarding multiple batches before computing a backward pass is
        # not possible, thus it is safe to store additional inputs here temporarily
        self.forward_inputs["attention_mask"] = attention_mask
        self.forward_inputs["encoder_attention_mask"] = encoder_attention_mask
        self.forward_inputs["head_mask"] = head_mask
        self.forward_inputs["cross_attn_head_mask"] = cross_attn_head_mask
        self.forward_inputs["past_key_values"] = past_key_values

    def set_device(self, device):
        self.device = device

    def get_forward_result(self, key):
        if key in ("hidden_states", "attentions", "cross_attentions"):
            batch_indexes = t.cat([x[0] for x in self.forward_results[key]])
            result = t.cat([x[1] for x in self.forward_results[key]])
            return result[t.argsort(batch_indexes)]
        else:
            # present_key_value_state
            batch_indexes = t.cat([x[0] for x in self.forward_results[key]])
            state_num = len(self.forward_results[key][0][1])
            result = [
                t.cat([x[1][state_idx] for x in self.forward_results[key]])
                for state_idx in range(state_num)
            ]
            sorted_indexes = t.argsort(batch_indexes)
            return tuple(res[sorted_indexes] for res in result)

    def clear_forward_results(self):
        for k in self.forward_results:
            self.forward_results[k] = []

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if isinstance(hidden_states, tuple):
            (
                hidden_states,
                position_bias,
                encoder_hidden_states,
                encoder_decoder_position_bias,
                batch_indexes,
            ) = hidden_states
            # batch dimension first
            layer_head_mask = (
                self.forward_inputs["head_mask"][self.index, batch_indexes]
                if self.forward_inputs["head_mask"] is not None
                else None
            )
            cross_attn_layer_head_mask = (
                self.forward_inputs["cross_attn_head_mask"][self.index, batch_indexes]
                if self.forward_inputs["cross_attn_head_mask"] is not None
                else None
            )
            past_key_value = (
                [
                    x[batch_indexes]
                    for x in self.forward_inputs["past_key_values"][self.index]
                ]
                if self.forward_inputs["past_key_values"] is not None
                else None
            )

            layer_outputs = super(T5BlockForPipeline, self).forward(
                hidden_states,
                self.forward_inputs["attention_mask"][batch_indexes]
                if self.forward_inputs["attention_mask"] is not None
                else None,
                position_bias if not is_none_tensor(position_bias) else None,
                encoder_hidden_states
                if not is_none_tensor(encoder_hidden_states)
                else None,
                self.forward_inputs["encoder_attention_mask"][batch_indexes]
                if self.forward_inputs["encoder_attention_mask"] is not None
                else None,
                encoder_decoder_position_bias
                if not is_none_tensor(encoder_decoder_position_bias)
                else None,
                layer_head_mask,
                cross_attn_layer_head_mask,
                past_key_value,
                **self.forward_flags,
            )

            # layer_outputs = hidden-states, key-value-states(None if not using cache),
            # position_bias, (self-attention weights),
            # (cross-attention position bias, if is decoder), (cross-attention weights, if is decoder)
            if self.forward_flags["use_cache"] is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            self.forward_results["hidden_states"].append(
                (batch_indexes, layer_outputs[0])
            )
            self.forward_results["present_key_value_state"].append(
                (batch_indexes, layer_outputs[1])
            )

            if self.forward_flags["output_attentions"]:
                self.forward_results["attentions"].append(
                    (batch_indexes, layer_outputs[3])
                )
                if self.is_decoder:
                    self.forward_results["cross_attentions"].append(
                        (batch_indexes, layer_outputs[5])
                    )

            if self.is_decoder and not is_none_tensor(encoder_hidden_states):
                encoder_decoder_position_bias = layer_outputs[
                    4 if self.forward_flags["output_attentions"] else 3
                ]
            result = (
                layer_outputs[0],
                layer_outputs[2],
                encoder_hidden_states,
                encoder_decoder_position_bias,
                batch_indexes,
            )
            return result
        else:
            return super(T5BlockForPipeline, self).forward(
                hidden_states,
                attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_decoder_position_bias,
                layer_head_mask,
                cross_attn_layer_head_mask,
                past_key_value,
                use_cache,
                output_attentions,
                return_dict,
            )


class T5StackForPipeline(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [
                T5BlockForPipeline(config, i, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        self.block_pipe = None
        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None, chunks=1):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = (
            "cpu"
            if "cpu" in self.device_map.keys()
            else "cuda:" + str(min(self.device_map.keys()))
        )
        self.last_device = "cuda:" + str(max(self.device_map.keys()))

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

        # Set device for each block
        for index, block in enumerate(self.block):
            block.set_device(
                [
                    device
                    for device, block_numbers in self.device_map.items()
                    if index in block_numbers
                ][0]
            )

        # Create pipe
        self.block_pipe = Pipe(
            nn.Sequential(*[b for b in self.block]),
            balance=[
                len(layer_numbers) for _, layer_numbers in self.device_map.items()
            ],
            devices=["cuda:" + str(k) for k in self.device_map.keys()],
            chunks=chunks,
            checkpoint="never",
        )

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.block_pipe = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            assert (
                self.embed_tokens is not None
            ), "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            past_key_values[0][0].shape[2] + seq_length
            if past_key_values is not None
            else seq_length
        )

        if use_cache is True:
            assert (
                self.is_decoder
            ), f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"
        if getattr(self.config, "gradient_checkpointing", False):
            assert (
                not self.model_parallel
            ), ":obj:`gradient_checkpointing` in configuration can only be set to `True` if model is not parallelized"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(
                inputs_embeds.device
            )
        if (
            self.is_decoder
            and encoder_attention_mask is None
            and encoder_hidden_states is not None
        ):
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size,
                encoder_seq_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, inputs_embeds.device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers
        )

        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        if self.model_parallel:
            # Begin forward through layers
            for layer_module in self.block:
                layer_module.set_forward_flags(
                    use_cache=use_cache, output_attentions=output_attentions
                )

            encoder_hidden_states = (
                encoder_hidden_states
                if encoder_hidden_states is not None
                else none_tensor(batch_size)
            )
            position_bias = none_tensor(batch_size)
            encoder_decoder_position_bias = none_tensor(batch_size)

            cache = {}
            for layer_module in self.block:
                dev = layer_module.device
                layer_module.set_forward_inputs(
                    attention_mask=move_or_use_existing_tensor(
                        cache, f"attention_mask_{dev}", extended_attention_mask, dev
                    ),
                    encoder_attention_mask=move_or_use_existing_tensor(
                        cache,
                        f"encoder_attention_mask_{dev}",
                        encoder_extended_attention_mask,
                        dev,
                    )
                    if encoder_extended_attention_mask is not None
                    else None,
                    head_mask=move_or_use_existing_tensor(
                        cache, f"head_mask_{dev}", head_mask, dev
                    )
                    if t.is_tensor(head_mask)
                    else None,
                    cross_attn_head_mask=move_or_use_existing_tensor(
                        cache, f"cross_attn_head_mask_{dev}", cross_attn_head_mask, dev
                    )
                    if t.is_tensor(cross_attn_head_mask)
                    else None,
                    past_key_values=tuple(
                        tuple(
                            move_or_use_existing_tensor(
                                cache, f"past_key_values_{i}_{dev}", xx, dev
                            )
                            for i, xx in enumerate(x)
                        )
                        for x in past_key_values
                    )
                    if past_key_values[0] is not None
                    else None,
                )

            outputs = self.block_pipe(
                (
                    # Always make sure that the first input requires gradient,
                    # Otherwise recompute stage will not be able to perform backward
                    hidden_states.to(self.first_device),
                    position_bias.to(self.first_device),
                    encoder_hidden_states.to(self.first_device),
                    encoder_decoder_position_bias.to(self.first_device),
                    t.range(0, batch_size - 1, dtype=t.long, device=self.first_device),
                )
            )
            hidden_states = outputs[0]
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)
            if use_cache:
                present_key_value_states = tuple(
                    tuple(
                        x.cpu()
                        for x in block.get_forward_result("present_key_value_state")
                    )
                    for block in self.block
                )
            if output_hidden_states:
                all_hidden_states = tuple(
                    block.get_forward_result("hidden_states").cpu()
                    for block in self.block
                )
            if output_attentions:
                all_attentions = tuple(
                    block.get_forward_result("attentions").cpu() for block in self.block
                )
                if self.is_decoder:
                    all_cross_attentions = tuple(
                        block.get_forward_result("cross_attentions").cpu()
                        for block in self.block
                    )
            for block in self.block:
                block.clear_forward_results()

        else:
            for i, (layer_module, past_key_value) in enumerate(
                zip(self.block, past_key_values)
            ):
                layer_head_mask = head_mask[i]
                cross_attn_layer_head_mask = cross_attn_head_mask[i]
                # Model parallel
                if self.model_parallel:
                    torch.cuda.set_device(hidden_states.device)
                    # Ensure that attention_mask is always on the same device as hidden_states
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(hidden_states.device)
                    if position_bias is not None:
                        position_bias = position_bias.to(hidden_states.device)
                    if encoder_hidden_states is not None:
                        encoder_hidden_states = encoder_hidden_states.to(
                            hidden_states.device
                        )
                    if encoder_extended_attention_mask is not None:
                        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                            hidden_states.device
                        )
                    if encoder_decoder_position_bias is not None:
                        encoder_decoder_position_bias = encoder_decoder_position_bias.to(
                            hidden_states.device
                        )
                    if layer_head_mask is not None:
                        layer_head_mask = layer_head_mask.to(hidden_states.device)
                    if cross_attn_layer_head_mask is not None:
                        cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(
                            hidden_states.device
                        )
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                if (
                    getattr(self.config, "gradient_checkpointing", False)
                    and self.training
                ):
                    if use_cache:
                        logger.warn(
                            "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                            "`use_cache=False`..."
                        )
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return tuple(module(*inputs, use_cache, output_attentions))

                        return custom_forward

                    layer_outputs = checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        extended_attention_mask,
                        position_bias,
                        encoder_hidden_states,
                        encoder_extended_attention_mask,
                        encoder_decoder_position_bias,
                        layer_head_mask,
                        cross_attn_layer_head_mask,
                        None,  # past_key_value is always None with gradient checkpointing
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=extended_attention_mask,
                        position_bias=position_bias,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        encoder_decoder_position_bias=encoder_decoder_position_bias,
                        layer_head_mask=layer_head_mask,
                        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )

                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, (self-attention position bias),
                # (self-attention weights), (cross-attention position bias), (cross-attention weights)
                if use_cache is False:
                    layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

                hidden_states, present_key_value_state = layer_outputs[:2]

                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention position bias),
                # (self-attention weights), (cross-attention position bias), (cross-attention weights)
                position_bias = layer_outputs[2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[
                        4 if output_attentions else 3
                    ]
                # append next layer key value states
                if use_cache:
                    present_key_value_states = present_key_value_states + (
                        present_key_value_state,
                    )

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[3],)
                    if self.is_decoder:
                        all_cross_attentions = all_cross_attentions + (
                            layer_outputs[5],
                        )

                    # Model Parallel: If it's the last layer for that device, put things on the next device
                if self.model_parallel:
                    for k, v in self.device_map.items():
                        if i == v[-1] and "cuda:" + str(k) != self.last_device:
                            hidden_states = hidden_states.to("cuda:" + str(k + 1))

            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)

            # Add last layer
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class T5ForConditionalGenerationForPipeline(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        del self.encoder
        del self.decoder

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5StackForPipeline(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5StackForPipeline(decoder_config, self.shared)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None, chunks=1):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map, chunks)
        self.decoder.parallelize(self.device_map, chunks)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True
