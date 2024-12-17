import MoonshineElementManager from "./elementManager";
import { MoonshineSettings } from "./constants";

// set the asset path to the CDN root since this script version is intended for CDN use only
MoonshineSettings.BASE_ASSET_PATH =
    "https://cdn.jsdelivr.net/npm/@usefulsensors/moonshine@latest/dist/";

var elementManager = new MoonshineElementManager("model/tiny");
elementManager.autoInjectElements();
elementManager.initControlElements();
