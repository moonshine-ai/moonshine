import numpy as np
import os
from pathlib import Path
import re
import tensorflow as tf


class MoonshineTFLite:
    def __init__(self, model_dir=os.path.join("base", "float")):
        self.preprocessor = tf.lite.Interpreter(os.path.join(model_dir, "preprocessor.tfl"))
        self.encoder = tf.lite.Interpreter(os.path.join(model_dir, "encoder.tfl"))
        self.decoder_initial = tf.lite.Interpreter(os.path.join(model_dir, "decoder_initial.tfl"))
        self.decoder = tf.lite.Interpreter(os.path.join(model_dir, "decoder.tfl"))

        self.preprocessor.allocate_tensors()
        self.encoder.allocate_tensors()
        self.decoder_initial.allocate_tensors()
        self.decoder.allocate_tensors()

    def _sort_by_name(self, names):
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [ atoi(c) for c in re.split(r'(\d+)', text) ]

        names.sort(key=natural_keys)
        return names

    def _sort_inputs(self, input_details):
        name_dict = {elem['name']: elem['index'] for elem in input_details}
        return [name_dict[key] for key in self._sort_by_name(list(name_dict.keys()))]

    def _rearrange_indices(self, arr):
        # Rearrange string sorted (0, 1, 10, 11, 12...) to index sorted (0, 1, 2, 3...).
        indices_rearrange = [int(j) for j in sorted([str(i) for i in range(len(arr))])]
        indices_rearrange = [indices_rearrange.index(i) for i in range(len(arr))]

        return [arr[i] for i in indices_rearrange]

    def _sort_outputs(self, output_details):
        name_dict = {elem['name']: elem['index'] for elem in output_details}

        names_out = [name for name in name_dict.keys() if 'StatefulPartitionedCall_1' in name]
        indices = [name_dict[key] for key in self._sort_by_name(names_out)]

        return self._rearrange_indices(indices)


    def _sort_outputs_decoder(self, output_details):
        name_dict = {elem['name']: elem['index'] for elem in output_details}

        names_x_attn = [name for name in name_dict.keys() if 'serving_default_keras_tensor' in name]
        indices_x_attn = [name_dict[key] for key in self._sort_by_name(names_x_attn)]

        # Since outputs are interleaved, there is no clean way to auto map them,
        # henice this horrible manual map.
        names_self_attn = [f'StatefulPartitionedCall_1:{d}' for d in [0, 1, 12, 28, 29, 32, 2, 5, 6, 9, 10, 14, 15, 18, 19, 22, 24]]
        indices_self_attn = [name_dict[key] for key in names_self_attn]

        # Logits should be first output.
        indices = [indices_self_attn[0]]

        # Interleave outputs since x attn and self attn kv caches are saved for each layer.
        for i in range(0, 16, 2):
            indices += indices_self_attn[i+1:i+3]
            indices += indices_x_attn[i:i+2]

        return indices


    def _invoke_tflite(self, interpreter, args, cached_decoder=False):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # super janky name-based sorting for inputs.
        input_indices = self._sort_inputs(input_details)

        # Resize inputs and allocate tensors.
        for idx, arg in enumerate(args):
            interpreter.resize_tensor_input(input_indices[idx], arg.shape)
        interpreter.allocate_tensors()

        # Set inputs and invoke.
        for idx, arg in enumerate(args):
            interpreter.set_tensor(input_indices[idx], arg)
        interpreter.invoke()

        # even jankier hybrid name and topological based sorting for outputs.
        if cached_decoder:
            output_indices = self._sort_outputs_decoder(output_details)
        else:
            output_indices = self._sort_outputs(output_details)
        return [interpreter.get_tensor(idx) for idx in output_indices]

    def __call__(self, audio, max_len=228):
        features = self._invoke_tflite(self.preprocessor, [audio])[0]
        embeddings = self._invoke_tflite(self.encoder, [features, np.array(features.shape[-2], dtype=np.int32)])[0]
        tokens = np.array([[1]], dtype=np.int32)
        output = tokens
        seq_len = np.array([1], dtype=np.int32)
        logits, *cache = self._invoke_tflite(self.decoder_initial, [tokens, seq_len, embeddings])
        for _ in range(max_len):
            tokens = np.argmax(logits, axis=-1).astype(np.int32)
            output = np.concatenate([output, tokens], axis=-1)
            if tokens[0, 0] == 2:
                break
            seq_len = seq_len + 1
            logits, *cache = self._invoke_tflite(self.decoder, [tokens, seq_len, embeddings] + cache, cached_decoder=True)

        return output


if __name__ == '__main__':

    import torchaudio
    import tokenizers
    import sys

    ASSETS_DIR = os.path.join(Path(__file__).parents[1], "moonshine", "assets")

    model = MoonshineTFLite()
    tokenizer_filename = os.path.join(ASSETS_DIR, "tokenizer.json")
    print(tokenizer_filename)
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_filename)

    if len(sys.argv) < 2:
        audio_filename = os.path.join(ASSETS_DIR, "beckett.wav")
    else:
        audio_filename = sys.argv[1]

    audio, sr = torchaudio.load(audio_filename)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = audio.numpy()

    tokens = model(audio)[0]
    text = tokenizer.decode(tokens).strip()
    print(text)