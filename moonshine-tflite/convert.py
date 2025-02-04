from moonshine import load_model
import os
import tensorflow as tf

def save_tfl(keras_model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(tflite_model)

for model_name in ["tiny", "base"]:
    
    model = load_model(f"moonshine/{model_name}")

    precision = "float"

    preprocessor_keras = model.preprocessor.preprocess
    encoder_keras = model.encoder.encoder
    decoder_initial_keras = model.decoder.uncached_call
    decoder_keras = model.decoder.cached_call

    save_tfl(preprocessor_keras, os.path.join(model_name, precision, "preprocessor.tfl"))
    save_tfl(encoder_keras, os.path.join(model_name, precision, "encoder.tfl"))
    save_tfl(decoder_initial_keras, os.path.join(model_name, precision, "decoder_initial.tfl"))
    save_tfl(decoder_keras, os.path.join(model_name, precision, "decoder.tfl"))
