// Helper script for downloading Moonshine ONNX models from HuggingFace for local development.
import * as fs from 'fs';
import * as hub from "@huggingface/hub";

const repo = { type: "model", name: "UsefulSensors/moonshine" };

var models = [
    "tiny",
    "base"
]

var layers = [
    "preprocess.ort",
    "encode.ort",
    "uncached_decode.ort",
    "cached_decode.ort"
]

console.log("Downloading Moonshine ONNX models from HuggingFace...")

models.forEach(model => {
    var dir = "public/moonshine/" + model
    if (!fs.existsSync(dir)){
        fs.mkdirSync(dir, { recursive: true });
    }
    layers.forEach(layer => {
        hub.downloadFile({ repo, path: "ort/" + model + "/" + layer }).then((file) => {
            file.arrayBuffer().then((buffer) => {
                fs.writeFile(dir + "/" + layer, Buffer.from(buffer), () => {
                    console.log("\tDownloaded " + model + "/" + layer + " successfully.") 
                });
            })
        })
        
    })
});
