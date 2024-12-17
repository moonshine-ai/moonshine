import * as ort from 'onnxruntime-web';
import llamaTokenizer from 'llama-tokenizer-js'
import { MoonshineSettings } from './constants';

function argMax(array) {
    return [].map.call(array, (x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

/**
 * Implements inference for moonshine models using the onnxruntime.
 */
export default class MoonshineModel {
    private modelURL: string
    private model: any

    /**
     * Create (but do not load (yet)) a new MoonshineModel for inference.
     * 
     * @param modelURL - a string (relative to {@link MoonshineSettings.BASE_ASSET_PATH}) where the .onnx model weights are located.
     * 
     * @remark Creating a MoonshineModel has the side effect of setting the path to the onnxruntime .wasm to the {@link MoonshineSettings.BASE_ASSET_PATH}
     */
    public constructor(modelURL: string) {
        this.modelURL = MoonshineSettings.BASE_ASSET_PATH + modelURL
        ort.env.wasm.wasmPaths = MoonshineSettings.BASE_ASSET_PATH
        this.model = {
            preprocess: undefined,
            encode: undefined,
            uncached_decode: undefined,
            cached_decode: undefined
        }
    }

    /**
     * Load the model weights.
     */
    public async loadModel() {
        // TODO choose the best sessionOption available
        // const sessionOption = { executionProviders: ['webgpu', 'webgl', 'wasm', 'cpu'] };
        const sessionOption = { executionProviders: ['wasm', 'cpu'] };

        this.model.preprocess = await ort.InferenceSession.create(
            this.modelURL + "/preprocess_quantized_weights.onnx", sessionOption)

        this.model.encode = await ort.InferenceSession.create(
            this.modelURL + "/encode_quantized_weights.onnx", sessionOption)

        this.model.uncached_decode = await ort.InferenceSession.create(
            this.modelURL + "/uncached_decode_quantized_weights.onnx", sessionOption)

        this.model.cached_decode = await ort.InferenceSession.create(
            this.modelURL + "/cached_decode_quantized_weights.onnx", sessionOption)
    }

    /**
     * Generate a transcription of the passed audio.
     * @param audio - a Float32Array containing raw audio samples from an audio source (e.g., a wav file, or a user's microphone)
     * @returns - a Promise that resolves with the generated transcription.
     */
    public async generate(audio: Float32Array) {
        if (this.model.preprocess && this.model.encode && this.model.uncached_decode
            && this.model.cached_decode) {
            const max_len = Math.trunc((audio.length / 16000) * 6)
            const preprocessed = await this.model.preprocess.run({
                args_0: new ort.Tensor("float32", audio, [1, audio.length])
            })
            const context = await this.model.encode.run({
                args_0: new ort.Tensor("float32", preprocessed["sequential"]["data"], preprocessed["sequential"]["dims"]),
                args_1: new ort.Tensor("int32", [preprocessed["sequential"]["dims"][1]], [1])
            })
            var seq_len = 1
            var layer_norm_key = ""
            for (const key in context) {
                if (key.startsWith("layer_norm")) {
                    layer_norm_key = key
                    break
                }
            }
            const uncached_decoded = await this.model.uncached_decode.run({
                args_0: new ort.Tensor("int32", [1], [1, 1]),
                args_1: new ort.Tensor("float32", context[layer_norm_key]["data"], context[layer_norm_key]["dims"]),
                args_2: new ort.Tensor("int32", [seq_len], [1])
            })
            var tokens = [1]
            var decode = uncached_decoded
            for (var i=0; i<max_len; i++) {
                var logits = decode["reversible_embedding"]["data"]
                const next_token = argMax(logits);
                if (next_token === 2) {
                    break;
                }
                tokens = tokens.concat([next_token])
                const inputs = [next_token]
                seq_len += 1
                const feed = {
                    args_0: new ort.Tensor("int32", inputs, [1, 1]),
                    args_1: new ort.Tensor("float32", context[layer_norm_key]["data"], context[layer_norm_key]["dims"]),
                    args_2: new ort.Tensor("int32", [seq_len], [1])
                }
                var j = 3
                Object.keys(decode).forEach(key => {
                    if (!key.startsWith("reversible")) {
                        feed["args_" + j] = decode[key];
                        j += 1
                    }
                });
                decode = await this.model.cached_decode.run(feed)
            }
            return llamaTokenizer.decode(tokens)
        }
        else {
            console.warn("MoonshineModel.generate(): Tried to call generate before the model was loaded.")
        }
    }
}
