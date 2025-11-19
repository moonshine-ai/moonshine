_MOONSHINE_FLAVORS = {
    "tiny": {"language": "English", "token_rate": 6},
    "tiny-ar": {"language": "Arabic", "token_rate": 13},
    "tiny-zh": {"language": "Chinese", "token_rate": 13},
    "tiny-ja": {"language": "Japanese", "token_rate": 13},
    "tiny-ko": {"language": "Korean", "token_rate": 13},
    "tiny-uk": {"language": "Ukrainian", "token_rate": 8},
    "tiny-vi": {"language": "Vietnamese", "token_rate": 13},
    "base": {"language": "English", "token_rate": 6},
    "base-es": {"language": "Spanish", "token_rate": 6},
}


def get_supported_languages():
    models = {k: v["language"] for k, v in _MOONSHINE_FLAVORS.items()}
    languages = sorted(list(set(models.values())))
    return {
        l: [k for k, v in _MOONSHINE_FLAVORS.items() if v["language"] == l]
        for l in languages
    }


def _get_onnx_weights(model_name, precision="float"):
    from huggingface_hub import hf_hub_download

    assert model_name in _MOONSHINE_FLAVORS, f'Unknown model "{model_name}"'

    repo = "UsefulSensors/moonshine"
    subfolder = f"onnx/merged/{model_name}/{precision}"

    return (
        hf_hub_download(repo, f"{x}.onnx", subfolder=subfolder)
        for x in ("encoder_model", "decoder_model_merged")
    )


class MoonshineOnnxModel(object):
    def __init__(
        self, models_dir=None, model_name=None, model_precision="float", token_rate=None, model_format="onnx"
    ):
        import onnxruntime

        # handle e.g., "moonshine/tiny" and "tiny"
        if model_name is not None:
            model_name = model_name.split("/")[-1]

        if models_dir is None:
            assert model_name is not None, (
                "model_name should be specified if models_dir is not"
            )
            encoder, decoder = self._load_weights_from_hf_hub(
                model_name, model_precision
            )
        else:
            encoder, decoder = [
                f"{models_dir}/{x}.{model_format}"
                for x in ("encoder_model", "decoder_model_merged")
            ]
            self.token_rate = token_rate

        if token_rate is None:
            self.token_rate = _MOONSHINE_FLAVORS[model_name]["token_rate"]

        self.encoder = onnxruntime.InferenceSession(encoder)
        self.decoder = onnxruntime.InferenceSession(decoder)

        self.encoder_input_names = [x.name for x in self.encoder.get_inputs()]
        self.decoder_input_names = [x.name for x in self.decoder.get_inputs()]

        if "tiny" in model_name:
            self.num_layers = 6
            self.num_key_value_heads = 8
            self.head_dim = 36
        elif "base" in model_name:
            self.num_layers = 8
            self.num_key_value_heads = 8
            self.head_dim = 52
        else:
            raise ValueError(f'Unknown model "{model_name}"')

        self.decoder_start_token_id = 1
        self.eos_token_id = 2

    def _load_weights_from_hf_hub(self, model_name, model_precision):
        return _get_onnx_weights(model_name, model_precision)

    def generate(self, audio, max_len=None):
        "audio has to be a numpy array of shape [1, num_audio_samples]"
        if max_len is None:
            max_len = int((audio.shape[-1] / 16_000) * self.token_rate)

        import numpy as np

        encoder_inputs = dict(input_values=audio)
        audio_attention_mask = np.ones_like(audio, dtype=np.int64)

        if "attention_mask" in self.encoder_input_names:
            encoder_inputs = dict(attention_mask=audio_attention_mask, **encoder_inputs)

        last_hidden_state = self.encoder.run(None, encoder_inputs)[0]

        past_key_values = {
            f"past_key_values.{i}.{a}.{b}": np.zeros(
                (0, self.num_key_value_heads, 1, self.head_dim), dtype=np.float32
            )
            for i in range(self.num_layers)
            for a in ("decoder", "encoder")
            for b in ("key", "value")
        }

        tokens = [self.decoder_start_token_id]
        input_ids = [tokens]
        for i in range(max_len):
            use_cache_branch = i > 0
            decoder_inputs = dict(
                input_ids=input_ids,
                encoder_hidden_states=last_hidden_state,
                use_cache_branch=[use_cache_branch],
                **past_key_values,
            )

            if "encoder_attention_mask" in self.decoder_input_names:
                decoder_inputs = dict(
                    encoder_attention_mask=audio_attention_mask, **decoder_inputs
                )

            logits, *present_key_values = self.decoder.run(None, decoder_inputs)
            next_token = logits[0, -1].argmax().item()
            tokens.append(next_token)
            if next_token == self.eos_token_id:
                break

            # Update values for next iteration
            input_ids = [[next_token]]
            for k, v in zip(past_key_values.keys(), present_key_values):
                if not use_cache_branch or "decoder" in k:
                    past_key_values[k] = v

        return [tokens]
