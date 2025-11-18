import keras
from keras import Model
from einops import rearrange


# See https://github.com/tensorflow/tensorflow/issues/47927#issuecomment-1499781808
# for why this workaround is needed to get consistent output ordering when exporting to TFLite.
def add_output_identity_dict(x):
    if not isinstance(x, list):
        x = [x]
    result = {}
    for layer in x:
        identity_layer = keras.layers.Identity()(layer)
        identity_layer.name = "output_identity_" + layer.name
        result[layer.name] = identity_layer
    return result


class AudioPreprocessor(object):
    def __init__(self, dim):
        inputs = keras.layers.Input([None, 1], name="audio_input")
        conv1 = keras.layers.Conv1D(
            filters=dim, kernel_size=127, strides=64, use_bias=False
        )
        tanh = keras.layers.Activation("tanh")
        group_norm = keras.layers.GroupNormalization(groups=1, axis=-1, epsilon=1e-5)
        conv2 = keras.layers.Conv1D(
            filters=2 * dim, kernel_size=7, strides=3, padding="valid"
        )
        gelu1 = keras.layers.Activation("gelu")
        conv3 = keras.layers.Conv1D(
            filters=dim, kernel_size=3, strides=2, padding="valid"
        )
        gelu2 = keras.layers.Activation("gelu")
        preprocess = keras.Sequential(
            [conv1, tanh, group_norm, conv2, gelu1, conv3, gelu2]
        )
        outputs = preprocess(inputs)
        outputs.name = "audio_features"
        outputs = add_output_identity_dict(outputs)
        self.preprocess = Model(inputs=inputs, outputs=outputs)

    def set_weights(self, weights):
        self.preprocess.set_weights(weights)

    def __call__(self, inputs):
        return self.preprocess(inputs)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = keras.ops.stack((-x2, x1), axis=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(t, freqs):
    rot_dim, seq_len, orig_dtype = (
        keras.ops.shape(freqs)[-1],
        keras.ops.shape(t)[-3],
        keras.ops.dtype(t),
    )
    freqs = freqs[-seq_len:, :]
    freqs = rearrange(freqs, "x y -> x 1 y")

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = t * keras.ops.cos(freqs) + rotate_half(t) * keras.ops.sin(freqs)
    out = keras.ops.concatenate((t, t_unrotated), axis=-1)

    return keras.ops.cast(out, orig_dtype)


class MHAWithRope(keras.layers.MultiHeadAttention):
    def call(self, query, value, key, rot_pos_emb):
        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)

        query = apply_rotary_pos_emb(query, rot_pos_emb)
        key = apply_rotary_pos_emb(key, rot_pos_emb)
        attention_output, _ = self._compute_attention(
            query=query,
            key=key,
            value=value,
            training=None,
        )

        output = self._output_dense(attention_output)
        return output

    def compute_output_spec(self, **kwargs):
        kwargs.pop("rot_pos_emb", None)
        return super(MHAWithRope, self).compute_output_spec(**kwargs)


class FFLinearGelu(object):
    def __init__(self, dim, ff_mult):
        ff = keras.Sequential(
            [
                keras.layers.Dense(dim * ff_mult, use_bias=True),
                keras.layers.Activation("gelu"),
                keras.layers.Dense(dim, use_bias=True),
            ]
        )
        inputs = keras.layers.Input([None, dim])
        outputs = ff(inputs)
        self.ff = Model(inputs=inputs, outputs=outputs)

    def set_weights(self, weights):
        self.ff.set_weights(weights)

    def __call__(self, x):
        return self.ff(x)


class FFSwiGLU(object):
    def __init__(self, dim, ff_mult):
        ff_proj = keras.layers.Dense(dim * ff_mult * 2, use_bias=True)
        ff_act = keras.layers.Activation("silu")
        ff_out = keras.layers.Dense(dim, use_bias=True)

        inputs = keras.layers.Input(shape=[None, dim])

        x = ff_proj(inputs)
        x, gate = keras.ops.split(x, 2, axis=-1)
        x = x * ff_act(gate)

        outputs = ff_out(x)
        self.ff_swiglu = Model(inputs=inputs, outputs=outputs)

    def set_weights(self, weights):
        self.ff_swiglu.set_weights(weights)

    def __call__(self, x):
        return self.ff_swiglu(x)


class EncoderLayer(object):
    def __init__(self, dim, inner_dim, n_head, ff_mult, ff_swiglu):
        norm1 = keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )

        attention = MHAWithRope(
            num_heads=n_head,
            key_dim=inner_dim // n_head,
            use_bias=False,
        )

        norm2 = keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )

        ff = FFSwiGLU(dim, ff_mult) if ff_swiglu else FFLinearGelu(dim, ff_mult)

        inputs = keras.layers.Input(shape=[None, dim])
        rot_pos_emb = keras.layers.Input(shape=[None, None], batch_size=1)
        rot_pos_emb = keras.ops.squeeze(rot_pos_emb)

        x = inputs
        _x = x
        x = norm1(x)
        x = attention(query=x, key=x, value=x, rot_pos_emb=rot_pos_emb)
        x = x + _x
        _x = x
        x = norm2(x)
        x = ff(x)
        outputs = x + _x
        self.encoder_layer = Model(inputs=[inputs, rot_pos_emb], outputs=outputs)

    def set_weights(self, weights):
        self.encoder_layer.set_weights(weights)

    def __call__(self, x, rot_pos_emb):
        return self.encoder_layer([x, rot_pos_emb])


class InvFreqInitializer(keras.initializers.Initializer):
    def __init__(self, dim, base):
        self.dim = dim
        self.base = base

    def __call__(self, shape, dtype=None):
        return 1.0 / (
            self.base
            ** (keras.ops.cast(keras.ops.arange(0, self.dim, 2), "float32") / self.dim)
        )


class RotaryEmbedding(Model):
    def __init__(self, dim, base=10000):
        super(RotaryEmbedding, self).__init__()
        # inv_freq will be float32 type
        self.inv_freq = self.add_weight(
            shape=(dim // 2,),
            initializer=InvFreqInitializer(dim, base),
            trainable=False,
        )

    def call(self, t):
        freqs = keras.ops.einsum(
            "i , j -> i j",
            keras.ops.cast(t, keras.ops.dtype(self.inv_freq)),
            self.inv_freq,
        )
        freqs = keras.ops.stack((freqs, freqs), axis=-1)
        return rearrange(freqs, "... d r -> ... (d r)")


class Arange(keras.layers.Layer):
    def call(self, inputs):
        return keras.ops.arange(inputs[0])

    def compute_output_spec(self, **kwargs):
        output_spec = keras.src.backend.KerasTensor((None,), dtype="float32")
        return output_spec


class Encoder(object):
    def __init__(self, n_layers, dim, inner_dim, n_head, ff_mult=4, ff_swiglu=False):
        rot_embed_dim = max(inner_dim // n_head // 2, 32)
        rot_pos_emb = RotaryEmbedding(rot_embed_dim)

        encoder_layers = [
            EncoderLayer(
                dim,
                inner_dim,
                n_head,
                ff_mult,
                ff_swiglu,
            )
            for _ in range(n_layers)
        ]
        final_norm = keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        inputs = keras.layers.Input(shape=[None, dim], name="audio_features")
        seq_len = keras.layers.Input(
            shape=[], batch_size=1, dtype="int32", name="seq_len"
        )
        pos_emb = rot_pos_emb(Arange()(inputs=seq_len))

        x = inputs
        for layer in encoder_layers:
            x = layer(x, pos_emb)
        outputs = final_norm(x)
        outputs.name = "last_hidden_state"
        outputs = add_output_identity_dict(outputs)
        self.encoder = Model(inputs=[inputs, seq_len], outputs=outputs)

    def set_weights(self, weights):
        self.encoder.set_weights(weights)

    def __call__(self, x, seq_len):
        return self.encoder([x, seq_len])


class MHACausalWithRope(keras.layers.MultiHeadAttention):
    def call(
        self,
        query,
        value,
        key,
        rot_pos_emb,
        value_cache=None,
        key_cache=None,
    ):
        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)

        query = apply_rotary_pos_emb(query, rot_pos_emb)
        key = apply_rotary_pos_emb(key, rot_pos_emb)

        if value_cache is not None:
            assert key_cache is not None, (
                "key_cache should not be None when value_cache is not"
            )

            key = keras.ops.concatenate((key_cache, key), axis=-3)
            value = keras.ops.concatenate((value_cache, value), axis=-3)

        causal_mask = self._compute_causal_mask(
            query, value, for_cache=value_cache is not None
        )

        attention_output, _ = self._compute_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=causal_mask,
            training=None,
        )

        output = self._output_dense(attention_output)
        return output, key, value

    def _compute_causal_mask(self, query, value=None, for_cache=False):
        if for_cache:
            assert value is not None, "value cannot be none if for_cache is True"
        v_seq_length = q_seq_length if value is None else keras.ops.shape(value)[1]
        q_seq_length = keras.ops.shape(query)[1]
        n_rows = v_seq_length if for_cache else q_seq_length
        ones_mask = keras.ops.ones((1, n_rows, v_seq_length), dtype="int32")
        row_index = keras.ops.cumsum(ones_mask, axis=-2)
        col_index = keras.ops.cumsum(ones_mask, axis=-1)
        mask = keras.ops.greater_equal(row_index, col_index)
        if for_cache:
            mask = mask[:, -q_seq_length:, :]
        return mask

    def compute_output_spec(self, **kwargs):
        kwargs.pop("rot_pos_emb", None)
        kwargs.pop("key_cache", None)
        kwargs.pop("value_cache", None)

        attention_spec = super(MHACausalWithRope, self).compute_output_spec(**kwargs)
        key_spec = keras.src.backend.KerasTensor(
            (None, None, self.num_heads, self.key_dim), dtype=self.compute_dtype
        )
        value_spec = keras.src.backend.KerasTensor(
            (None, None, self.num_heads, self.value_dim), dtype=self.compute_dtype
        )

        return attention_spec, key_spec, value_spec


class MHAPrecomputedKV(keras.layers.MultiHeadAttention):
    def call(self, query, value, key, key_cache=None, value_cache=None):
        query = self._query_dense(query)
        if key_cache is None:
            assert value_cache is None, "Both key and value cache have to be None"
            key = self.key_dense(key)
            value = self.value_dense(value)
        else:
            key = key_cache
            value = value_cache

        attention_output, _ = self._compute_attention(
            query=query,
            key=key,
            value=value,
            training=None,
        )

        output = self._output_dense(attention_output)
        if key_cache is None:
            return output, key, value
        return output

    def compute_output_spec(self, **kwargs):
        key_cache = kwargs.pop("key_cache", None)
        value_cache = kwargs.pop("value_cache", None)
        attention_spec = super(MHAPrecomputedKV, self).compute_output_spec(**kwargs)
        if key_cache is None:
            key_spec = keras.src.backend.KerasTensor(
                (None, None, self.num_heads, self.key_dim), dtype=self.compute_dtype
            )
            value_spec = keras.src.backend.KerasTensor(
                (None, None, self.num_heads, self.value_dim), dtype=self.compute_dtype
            )
            return attention_spec, key_spec, value_spec
        return attention_spec


class DecoderLayer(object):
    def __init__(self, dim, inner_dim, n_head, ff_mult, ff_swiglu):
        self.norm1 = keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        self.self_attention = MHACausalWithRope(
            num_heads=n_head,
            key_dim=inner_dim // n_head,
            use_bias=False,
        )
        self.norm2 = keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        self.cross_attention = MHAPrecomputedKV(
            num_heads=n_head,
            key_dim=inner_dim // n_head,
            use_bias=False,
        )
        self.norm3 = keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        self.ff = FFSwiGLU(dim, ff_mult) if ff_swiglu else FFLinearGelu(dim, ff_mult)

        self.uncached_call = self.get_uncached_call(dim)
        self.cached_call = self.get_cached_call(dim, inner_dim // n_head, n_head)

    def get_uncached_call(self, dim):
        inputs = keras.layers.Input(shape=[None, dim])
        context = keras.layers.Input(shape=[None, dim])
        rot_pos_emb = keras.layers.Input(shape=[None, None], batch_size=1)
        rot_pos_emb = keras.ops.squeeze(rot_pos_emb)

        x = inputs
        _x = x
        x = self.norm1(x)

        x, cache_k, cache_v = self.self_attention(
            query=x,
            key=x,
            value=x,
            rot_pos_emb=rot_pos_emb,
            value_cache=None,
            key_cache=None,
        )
        x = x + _x

        _x = x
        x = self.norm2(x)
        x, x_attn_cache_k, x_attn_cache_v = self.cross_attention(
            query=x,
            key=context,
            value=context,
            key_cache=None,
            value_cache=None,
        )
        x = x + _x

        _x = x
        x = self.norm3(x)
        x = self.ff(x)
        x = x + _x

        return Model(
            inputs=[inputs, context, rot_pos_emb],
            outputs=[x, cache_k, cache_v, x_attn_cache_k, x_attn_cache_v],
        )

    def get_cached_call(self, dim, key_dim, n_head):
        inputs = keras.layers.Input(shape=[None, dim])
        context = keras.layers.Input(shape=[None, dim])
        cache_k = keras.layers.Input(shape=[None, n_head, key_dim])
        cache_v = keras.layers.Input(shape=[None, n_head, key_dim])
        x_attn_cache_k = keras.layers.Input(shape=[None, n_head, key_dim])
        x_attn_cache_v = keras.layers.Input(shape=[None, n_head, key_dim])
        rot_pos_emb = keras.layers.Input(shape=[None, None], batch_size=1)
        rot_pos_emb = keras.ops.squeeze(rot_pos_emb)

        x = inputs
        _x = x
        x = self.norm1(x)

        x, new_cache_k, new_cache_v = self.self_attention(
            query=x,
            key=x,
            value=x,
            rot_pos_emb=rot_pos_emb,
            key_cache=cache_k,
            value_cache=cache_v,
        )
        x = x + _x

        _x = x
        x = self.norm2(x)
        x = self.cross_attention(
            query=x,
            key=context,
            value=context,
            key_cache=x_attn_cache_k,
            value_cache=x_attn_cache_v,
        )
        x = x + _x

        _x = x
        x = self.norm3(x)
        x = self.ff(x)
        x = x + _x

        return Model(
            inputs=[
                inputs,
                context,
                cache_k,
                cache_v,
                x_attn_cache_k,
                x_attn_cache_v,
                rot_pos_emb,
            ],
            outputs=[x, new_cache_k, new_cache_v],
        )


class ReversibleEmbedding(keras.layers.Embedding):
    def call(self, inputs, reverse=False):
        if reverse:
            kernel = keras.ops.transpose(keras.ops.convert_to_tensor(self.embeddings))
            logits = keras.ops.matmul(inputs, kernel)
            return logits

        return super().call(inputs)

    def compute_output_spec(self, inputs, reverse=False):
        output_shape = list(inputs.shape)
        if reverse:
            output_shape[-1] = self.input_dim
        else:
            output_shape += [self.output_dim]
        return keras.KerasTensor(output_shape, dtype=self.compute_dtype)


class Decoder(object):
    def __init__(
        self, n_layers, dim, inner_dim, n_head, vocab_size, ff_mult=4, ff_swiglu=True
    ):
        self.embedding_layer = ReversibleEmbedding(vocab_size, dim)
        self.decoder_layers = [
            DecoderLayer(dim, inner_dim, n_head, ff_mult, ff_swiglu)
            for _ in range(n_layers)
        ]
        self.post_norm = keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )

        rot_embed_dim = max(inner_dim // n_head // 2, 32)
        self.uncached_call = self.get_uncached_call(dim, rot_embed_dim)
        self.cached_call = self.get_cached_call(
            dim, rot_embed_dim, inner_dim // n_head, n_head, n_layers
        )

    def get_uncached_call(self, dim, rot_embed_dim):
        inputs = keras.layers.Input(shape=[None], dtype="int32", name="input_ids")
        seq_len = keras.layers.Input(
            shape=[], batch_size=1, dtype="int32", name="seq_len"
        )
        context = keras.layers.Input(
            shape=[None, dim], dtype="float32", name="last_hidden_state"
        )
        rot_pos_emb = RotaryEmbedding(rot_embed_dim)

        x = inputs
        x = self.embedding_layer(x)
        rot_pos_emb = rot_pos_emb(Arange()(inputs=seq_len))

        outputs = []
        for i, d in enumerate(self.decoder_layers):
            x, cache_k, cache_v, x_attn_cache_k, x_attn_cache_v = d.uncached_call(
                [x, context, rot_pos_emb]
            )
            cache_k.name = f"cache_k_{i}"
            cache_v.name = f"cache_v_{i}"
            x_attn_cache_k.name = f"x_attn_cache_k_{i}"
            x_attn_cache_v.name = f"x_attn_cache_v_{i}"
            outputs.extend([cache_k, cache_v, x_attn_cache_k, x_attn_cache_v])
        x = self.post_norm(x)

        logits = self.embedding_layer(x, reverse=True)
        logits.name = "logits"
        return Model(inputs=[inputs, context, seq_len], outputs=[logits] + outputs)

    def get_cached_call(self, dim, rot_embed_dim, key_dim, n_head, n_layers):
        inputs = keras.layers.Input(shape=[None], dtype="int32", name="input_ids")
        seq_len = keras.layers.Input(
            shape=[], batch_size=1, dtype="int32", name="seq_len"
        )
        context = keras.layers.Input(
            shape=[None, dim], dtype="float32", name="last_hidden_state"
        )
        rot_pos_emb = RotaryEmbedding(rot_embed_dim)

        cache = [
            [
                keras.layers.Input(
                    shape=[None, n_head, key_dim], dtype="float32", name=f"cache_k_{i}"
                ),
                keras.layers.Input(
                    shape=[None, n_head, key_dim], dtype="float32", name=f"cache_v_{i}"
                ),
                keras.layers.Input(
                    shape=[None, n_head, key_dim],
                    dtype="float32",
                    name=f"x_attn_cache_k_{i}",
                ),
                keras.layers.Input(
                    shape=[None, n_head, key_dim],
                    dtype="float32",
                    name=f"x_attn_cache_v_{i}",
                ),
            ]
            for i in range(n_layers)
        ]
        cache = sum(cache, [])

        x = inputs
        x = self.embedding_layer(x)
        rot_pos_emb = rot_pos_emb(Arange()(inputs=seq_len))

        outputs = []
        for i, d in enumerate(self.decoder_layers):
            x, new_cache_k, new_cache_v = d.cached_call(
                [
                    x,
                    context,
                    cache[4 * i + 0],
                    cache[4 * i + 1],
                    cache[4 * i + 2],
                    cache[4 * i + 3],
                    rot_pos_emb,
                ]
            )
            new_cache_k.name = f"cache_k_{i}"
            new_cache_v.name = f"cache_v_{i}"
            outputs.extend(
                [new_cache_k, new_cache_v, cache[4 * i + 2], cache[4 * i + 3]]
            )

        x = self.post_norm(x)

        logits = self.embedding_layer(x, reverse=True)
        logits.name = "logits"

        return Model(
            inputs=[inputs, context, seq_len] + cache, outputs=[logits] + outputs
        )

    def set_weights(self, weights):
        self.uncached_call.set_weights(weights)


class Moonshine(object):
    def __init__(
        self,
        dim,
        inner_dim,
        n_head,
        enc_n_layers,
        dec_n_layers,
        enc_ff_mult=4,
        dec_ff_mult=4,
        enc_ff_swiglu=False,
        dec_ff_swiglu=True,
        vocab_size=32768,
    ):
        self.preprocessor = AudioPreprocessor(dim)
        self.encoder = Encoder(
            enc_n_layers, dim, inner_dim, n_head, enc_ff_mult, enc_ff_swiglu
        )
        self.decoder = Decoder(
            dec_n_layers, dim, inner_dim, n_head, vocab_size, dec_ff_mult, dec_ff_swiglu
        )
        self.dim = dim
        self.inner_dim = inner_dim
        self.n_head = n_head
        self.enc_n_layers = enc_n_layers
        self.dec_n_layers = dec_n_layers

    def _load_weights(self, preprocessor_weights, encoder_weights, decoder_weights):
        self.preprocessor.preprocess.load_weights(preprocessor_weights)
        self.encoder.encoder.load_weights(encoder_weights)
        self.decoder.uncached_call.load_weights(decoder_weights)

    def generate(self, audio, max_len=None):
        if max_len is None:
            # max 6 tokens per second of audio
            max_len = int((audio.shape[-1] / 16_000) * 6)
        audio_features = self.preprocessor(audio)["audio_features"]

        last_hidden_state = self.encoder(
            audio_features, keras.ops.convert_to_tensor([audio_features.shape[-2]])
        )["last_hidden_state"]

        tokens = keras.ops.convert_to_tensor([[1]])
        seq_len = keras.ops.convert_to_tensor([1])
        logits, *cache = self.decoder.uncached_call(
            [tokens, last_hidden_state, seq_len]
        )
        output = tokens
        for _ in range(max_len):
            tokens = keras.ops.argmax(logits, axis=-1)
            output = keras.ops.concatenate([output, tokens], axis=-1)
            if tokens[0, 0] == 2:
                break
            seq_len = seq_len + 1
            logits, *cache = self.decoder.cached_call(
                [tokens, last_hidden_state, seq_len] + cache
            )

        return keras.ops.convert_to_numpy(output)


def _get_weights(model_name):
    from huggingface_hub import hf_hub_download

    repo = "UsefulSensors/moonshine"

    return (
        hf_hub_download(repo, f"{x}.weights.h5", subfolder=model_name)
        for x in ("preprocessor", "encoder", "decoder")
    )


def load_model(model_name):
    if model_name == "moonshine/base":
        model = Moonshine(416, 416, 8, 8, 8)
        model._load_weights(*_get_weights("base"))
        return model
    if model_name == "moonshine/tiny":
        model = Moonshine(288, 288, 8, 6, 6)
        model._load_weights(*_get_weights("tiny"))
        return model
    assert False, f"{model_name} not a valid moonshine model"
