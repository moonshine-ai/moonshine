import os
import sys
import onnxruntime

# TODO(guy): review this change.
# import moonshine
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from transcribe import load_audio, load_tokenizer


class MoonshineOnnxModel(object):
    def __init__(self, models_dir):
        self.preprocess = onnxruntime.InferenceSession(f"{models_dir}/preprocess.onnx")
        self.encode = onnxruntime.InferenceSession(f"{models_dir}/encode.onnx")
        self.uncached_decode = onnxruntime.InferenceSession(
            f"{models_dir}/uncached_decode.onnx"
        )
        self.cached_decode = onnxruntime.InferenceSession(
            f"{models_dir}/cached_decode.onnx"
        )
        # TODO(guy): review this change.
        # self.tokenizer = moonshine.load_tokenizer()
        self.tokenizer = load_tokenizer()

    def generate(self, audio, max_len=None):
        # TODO(guy): review this change.
        # audio = moonshine.load_audio(audio, return_numpy=True)
        audio = load_audio(audio, return_numpy=True)
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
