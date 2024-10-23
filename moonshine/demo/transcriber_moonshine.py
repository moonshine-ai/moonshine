"""Moonshine model transcriber."""
import time

import numpy as np
import tokenizers

from moonshine import load_model, ASSETS_DIR

class TranscriberMoonshine(object):
    def __init__(self, model_name="moonshine/tiny", rate=16000):
        if rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz.")
        self.model = load_model(model_name=model_name)
        self.rate = rate
        tokenizer_file = ASSETS_DIR / "tokenizer.json"
        self.tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_file))

        self.inference_secs = 0
        self.number_inferences = 0
        self.speech_secs = 0
        self.__call__(np.zeros(int(rate)))  # Warmup.

    def __call__(self, speech):
        """Returns string containing Moonshine transcription of speech."""
        self.number_inferences += 1
        self.speech_secs += len(speech) / self.rate
        start_time = time.time()

        tokens = self.model.generate(speech[np.newaxis, :])
        text = self.tokenizer.decode_batch(tokens)[0]

        self.inference_secs += time.time() - start_time
        return text
