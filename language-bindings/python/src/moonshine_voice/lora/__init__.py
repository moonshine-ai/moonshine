"""LoRA domain adaptation for Moonshine Streaming.

This submodule is an opt-in training path. ``import moonshine_voice`` does not
load it, and the default inference wheel does not install PyTorch or
Transformers. Install the extra, then either::

    pip install 'moonshine-voice[finetune]'
    python -m moonshine_voice.lora --dataset atcosim --output-dir ./lora_atc

or::

    from moonshine_voice.lora import fit_adapter, train_adapter
"""

from moonshine_voice.lora._deps import require_lora_deps

_EXPORTS = {
    "AtcosimIndex": ("moonshine_voice.lora.data", "AtcosimIndex"),
    "LoRALinear": ("moonshine_voice.lora.adapter", "LoRALinear"),
    "SAMPLE_RATE": ("moonshine_voice.lora.data", "SAMPLE_RATE"),
    "UwbAtccIndex": ("moonshine_voice.lora.data", "UwbAtccIndex"),
    "Utterance": ("moonshine_voice.lora.manifest", "Utterance"),
    "add_lora": ("moonshine_voice.lora.adapter", "add_lora"),
    "atcosim_source": ("moonshine_voice.lora.data", "atcosim_source"),
    "build_cache": ("moonshine_voice.lora.data", "build_cache"),
    "corpus_wer": ("moonshine_voice.lora.train", "corpus_wer"),
    "default_lr": ("moonshine_voice.lora.train", "default_lr"),
    "decode_atco2": ("moonshine_voice.lora.data", "decode_atco2"),
    "decode_atcosim": ("moonshine_voice.lora.data", "decode_atcosim"),
    "decode_uwb_atcc": ("moonshine_voice.lora.data", "decode_uwb_atcc"),
    "encode_text": ("moonshine_voice.lora.train", "encode_text"),
    "english_normalizer": ("moonshine_voice.lora.train", "english_normalizer"),
    "export_checkpoint": ("moonshine_voice.lora.export", "export_checkpoint"),
    "fit_adapter": ("moonshine_voice.lora.train", "fit_adapter"),
    "hours_of": ("moonshine_voice.lora.data", "hours_of"),
    "index_atco2": ("moonshine_voice.lora.data", "index_atco2"),
    "index_atcosim": ("moonshine_voice.lora.data", "index_atcosim"),
    "index_uwb_atcc": ("moonshine_voice.lora.data", "index_uwb_atcc"),
    "librispeech_eval": ("moonshine_voice.lora.data", "librispeech_eval"),
    "load_manifest": ("moonshine_voice.lora.manifest", "load_manifest"),
    "open_blob": ("moonshine_voice.lora.data", "open_blob"),
    "prepare_adaptation": ("moonshine_voice.lora.adapter", "prepare_adaptation"),
    "replay_source": ("moonshine_voice.lora.data", "replay_source"),
    "sample_indices": ("moonshine_voice.lora.train", "sample_indices"),
    "session_disjoint_train": ("moonshine_voice.lora.data", "session_disjoint_train"),
    "train_adapter": ("moonshine_voice.lora.train", "train_adapter"),
    "transcribe": ("moonshine_voice.lora.train", "transcribe"),
    "uwb_atcc_source": ("moonshine_voice.lora.data", "uwb_atcc_source"),
    "uwb_session": ("moonshine_voice.lora.data", "uwb_session"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Manifest helpers are stdlib-only; everything else needs the extra.
    module_name, attr = _EXPORTS[name]
    if module_name != "moonshine_voice.lora.manifest":
        require_lora_deps()
    from importlib import import_module

    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value
