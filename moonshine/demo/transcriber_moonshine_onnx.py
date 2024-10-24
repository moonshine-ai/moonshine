"""Moonshine model transcriber Onnx version."""
import os
import sys
import time

import numpy as np
import tokenizers

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tools.onnx_model import MoonshineOnnxModel


class TranscriberMoonshine(object):
    def __init__(self, models_dir=None, rate=16000):
        if models_dir is None:
            raise ValueError("Provide full path to Moonshine ONNX model.")
        if rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz.")
        self.model = MoonshineOnnxModel(models_dir=models_dir)
        self.rate = rate
        assets_dir = f"{os.path.join(os.path.dirname(__file__), '..', 'assets')}"
        tokenizer_file = f"{assets_dir}{os.sep}tokenizer.json"
        self.tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_file))

        self.inference_secs = 0
        self.number_inferences = 0
        self.speech_secs = 0
        self.__call__(np.zeros(int(rate), dtype=np.float32))  # Warmup.

    def __call__(self, speech):
        """Returns string containing Moonshine transcription of speech."""
        self.number_inferences += 1
        self.speech_secs += len(speech) / self.rate
        start_time = time.time()

        tokens = self.model.generate(speech[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch(tokens)[0]

        self.inference_secs += time.time() - start_time
        return text
