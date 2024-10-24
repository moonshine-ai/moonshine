def _get_onnx_weights(model_name):
    from huggingface_hub import hf_hub_download

    repo = "UsefulSensors/moonshine"

    return (
        hf_hub_download(repo, f"{x}.onnx", subfolder=f"onnx/{model_name}")
        for x in ("preprocess", "encode", "uncached_decode", "cached_decode")
    )


class MoonshineOnnxModel(object):
    def __init__(self, models_dir=None, model_name=None):
        import onnxruntime

        if models_dir is None:
            assert (
                model_name is not None
            ), "model_name should be specified if models_dir is not"
            preprocess, encode, uncached_decode, cached_decode = (
                self._load_weights_from_hf_hub(model_name)
            )
        else:
            preprocess, encode, uncached_decode, cached_decode = [
                f"{models_dir}/{x}.onnx"
                for x in ["preprocess", "encode", "uncached_decode", "cached_decode"]
            ]
        self.preprocess = onnxruntime.InferenceSession(preprocess)
        self.encode = onnxruntime.InferenceSession(encode)
        self.uncached_decode = onnxruntime.InferenceSession(uncached_decode)
        self.cached_decode = onnxruntime.InferenceSession(cached_decode)

    def _load_weights_from_hf_hub(self, model_name):
        model_name = model_name.split("/")[-1]
        return _get_onnx_weights(model_name)

    def generate(self, audio, max_len=None):
        "audio has to be a numpy array of shape [1, num_audio_samples]"
        if max_len is None:
            # max 6 tokens per second of audio
            max_len = int((audio.shape[-1] / 16_000) * 6)
        preprocessed = self.preprocess.run([], dict(args_0=audio))[0]
        seq_len = [preprocessed.shape[-2]]

        context = self.encode.run([], dict(args_0=preprocessed, args_1=seq_len))[0]
        inputs = [[1]]
        seq_len = [1]

        tokens = [1]
        logits, *cache = self.uncached_decode.run(
            [], dict(args_0=inputs, args_1=context, args_2=seq_len)
        )
        for i in range(max_len):
            next_token = logits.squeeze().argmax()
            tokens.extend([next_token])
            if next_token == 2:
                break

            seq_len[0] += 1
            inputs = [[next_token]]
            logits, *cache = self.cached_decode.run(
                [],
                dict(
                    args_0=inputs,
                    args_1=context,
                    args_2=seq_len,
                    **{f"args_{i+3}": x for i, x in enumerate(cache)},
                ),
            )
        return [tokens]
