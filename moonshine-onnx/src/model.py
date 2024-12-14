def _get_onnx_weights(model_name):
    from huggingface_hub import hf_hub_download
    
    if model_name not in ["tiny", "base"]:
        raise ValueError(f"Unknown model \"{model_name}\"")
    repo = f"onnx-community/moonshine-{model_name}-ONNX"

    return (
        hf_hub_download(repo, f"{x}.onnx", subfolder="onnx")
        for x in ("encoder_model", "decoder_model_merged")
    )


class MoonshineOnnxModel(object):
    def __init__(self, models_dir=None, model_name=None):
        import onnxruntime

        if models_dir is None:
            assert (
                model_name is not None
            ), "model_name should be specified if models_dir is not"
            encoder, decoder = (
                self._load_weights_from_hf_hub(model_name)
            )
        else:
            encoder, decoder = [
                f"{models_dir}/{x}.onnx"
                for x in ("encoder_model", "decoder_model_merged")
            ]
        self.encoder = onnxruntime.InferenceSession(encoder)
        self.decoder = onnxruntime.InferenceSession(decoder)

        if 'tiny' in model_name:
            self.num_layers = 6
            self.num_key_value_heads = 8
            self.head_dim = 36
        elif 'base' in model_name:
            self.num_layers = 8
            self.num_key_value_heads = 8
            self.head_dim = 52
        else:
            raise ValueError(f"Unknown model \"{model_name}\"")

        self.decoder_start_token_id = 1
        self.eos_token_id = 2

    def _load_weights_from_hf_hub(self, model_name):
        model_name = model_name.split("/")[-1]
        return _get_onnx_weights(model_name)

    def generate(self, audio, max_len=None):
        "audio has to be a numpy array of shape [1, num_audio_samples]"
        if max_len is None:
            # max 6 tokens per second of audio
            max_len = int((audio.shape[-1] / 16_000) * 6)

        import numpy as np

        last_hidden_state = self.encoder.run(None, dict(input_values=audio))[0]

        past_key_values = {
            f"past_key_values.{i}.{a}.{b}": np.zeros((0, self.num_key_value_heads, 1, self.head_dim), dtype=np.float32)
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
            logits, *present_key_values = self.decoder.run(None, decoder_inputs)
            next_token = logits[0, -1].argmax().item()
            tokens.append(next_token)
            if next_token == self.eos_token_id:
                break

            # Update values for next iteration
            input_ids = [[next_token]]
            for k, v in zip(past_key_values.keys(), present_key_values):
                if not use_cache_branch or 'decoder' in k:
                    past_key_values[k] = v

        return [tokens]
