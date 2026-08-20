# Training a domain adapter

The recipe, ATCOSIM phraseology example, real-VHF (`uwb_atcc`) example, dataset
format, export steps, and pitfalls are in
[Domain Customization](../../../docs/models/domain-customization.md#retraining).

```bash
pip install 'moonshine-voice[finetune]'
python -m moonshine_voice.lora --dataset atcosim --output-dir ./lora_atc
```

`[lora]` is the same extra. `moonshine-voice finetune` is the same command.
