"""Train a decoder-only LoRA adapter on a Moonshine Streaming checkpoint.

Imported only after ``require_lora_deps()``. The CLI lives in ``__main__.py``
so ``--help`` does not load PyTorch.
"""

from __future__ import annotations

import json
import math
import time
from argparse import Namespace
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

from moonshine_voice.lora.adapter import (
    adapter_parameters,
    adapter_state,
    add_lora,
    freeze_backbone,
    load_adapter_state,
    merge_and_restore,
)
from moonshine_voice.lora.data import (
    SAMPLE_RATE,
    atcosim_source,
    build_cache,
    decode_atcosim,
    file_source,
    hours_of,
    index_atcosim,
    librispeech_eval,
    open_blob,
    replay_source,
)
from moonshine_voice.lora.manifest import (
    Utterance,
    apply_text_mode,
    choose_text_mode,
    load_manifest,
)

BOS, EOS, PAD = 1, 2, 0
FRONTEND_STRIDE = 80


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _amp_dtype(device: str):
    if device != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def encode_text(processor, text: str) -> List[int]:
    return (
        [BOS]
        + processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        + [EOS]
    )


def make_batches(entries, indices, batch_size):
    ordered = sorted(indices, key=lambda i: entries[i]["samples"])
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


def collate(entries, blob, indices, device):
    rows = [entries[i] for i in indices]
    width = int(math.ceil(max(r["samples"] for r in rows) / FRONTEND_STRIDE) * FRONTEND_STRIDE)
    text_width = max(len(r["tokens"]) for r in rows)
    src = torch.zeros(len(rows), width)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    dst = torch.zeros(len(rows), text_width, dtype=torch.long)
    for i, row in enumerate(rows):
        chunk = np.asarray(blob[row["offset"] : row["offset"] + row["samples"]])
        src[i, : row["samples"]] = torch.from_numpy(chunk.astype(np.float32) / 32768.0)
        mask[i, : row["samples"]] = 1
        dst[i, : len(row["tokens"])] = torch.tensor(row["tokens"])
    return src.to(device), mask.to(device), dst.to(device)


def batch_loss(model, src, mask, dst, amp_dtype):
    # Shift explicitly rather than passing labels=: MoonshineStreaming
    # right-shifts labels into decoder_input_ids, and Transformers <=5.14 then
    # shifted them again. A pretrained model scores ~2.2 with this alignment
    # and ~10 with a wrong one.
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model(
            input_values=src,
            attention_mask=mask,
            decoder_input_ids=dst,
            use_cache=False,
        )
    return F.cross_entropy(
        out.logits[:, :-1].float().transpose(1, 2), dst[:, 1:], ignore_index=PAD
    )


def tail_split(entries, hours):
    """Held-out slice off the tail; training takes a prefix, so they cannot overlap."""
    held, taken = [], 0.0
    for i in range(len(entries) - 1, -1, -1):
        if taken >= hours * 3600:
            break
        held.append(i)
        taken += entries[i]["samples"] / SAMPLE_RATE
    return held, taken


_NORMALIZER = None


def english_normalizer():
    """Whisper English text normalizer, loaded once.

    Lowercases, strips punctuation, and expands numbers so WER compares words
    rather than typography. Spelled-out digits become joined digits; hyphenated
    numerals stay as separate tokens, which is why ATCOSIM's number convention
    dominates the baseline error rate.
    """
    global _NORMALIZER
    if _NORMALIZER is None:
        import json as json_mod

        from huggingface_hub import hf_hub_download
        from transformers.models.whisper.english_normalizer import (
            EnglishTextNormalizer,
        )

        _NORMALIZER = EnglishTextNormalizer(
            json_mod.load(
                open(hf_hub_download("openai/whisper-tiny", "normalizer.json"))
            )
        )
    return _NORMALIZER


def transcribe(model, processor, waves, device, batch_size=16, max_new_tokens=96):
    order = sorted(range(len(waves)), key=lambda i: len(waves[i]))
    texts = [None] * len(waves)
    model.eval()
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        inputs = processor(
            [waves[i] for i in chunk], sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        for i, text in zip(chunk, processor.batch_decode(ids, skip_special_tokens=True)):
            texts[i] = text
    return texts


def corpus_wer(refs, hyps):
    import jiwer

    normalize = english_normalizer()
    refs_n = [normalize(r) for r in refs]
    hyps_n = [normalize(h) for h in hyps]
    keep = [i for i, r in enumerate(refs_n) if r.strip()]
    return jiwer.wer([refs_n[i] for i in keep], [hyps_n[i] for i in keep]) * 100


def sample_indices(n, limit, seed=0):
    """All of ``n``, or a fixed random subset when ``limit`` is set."""
    if limit is None or limit >= n:
        return list(range(n))
    return sorted(np.random.default_rng(seed).choice(n, limit, replace=False).tolist())


def _load_train_rows(args) -> tuple:
    """Return (train_rows, eval_rows, domain_name, source_builder)."""
    if args.dataset == "atcosim":
        indexed = index_atcosim()
        train_pool, scored = indexed.train, indexed.scored
        print(
            f"ATCOSIM speaker-disjoint train: {len(train_pool)} utts / "
            f"{hours_of(train_pool):.2f} h, speakers "
            f"{sorted({r.speaker for r in train_pool})}"
        )
        print(
            f"ATCOSIM scored: {len(scored)} utts / {hours_of(scored):.2f} h, "
            f"speakers {sorted({r.speaker for r in scored})}"
        )
        text_mode = choose_text_mode([r.text for r in train_pool], args.text_mode)
        train_hours = args.train_hours if args.train_hours is not None else 2.0

        def source(hours, pool=train_pool, mode=text_mode):
            return atcosim_source(pool, hours, mode)

        return train_pool, scored, "atcosim", source, text_mode, train_hours

    if not args.train_manifest:
        raise SystemExit("pass --train-manifest PATH or --dataset atcosim")
    rows = load_manifest(args.train_manifest, args.data_root)
    missing = [r for r in rows if not r.audio]
    if missing:
        raise SystemExit(
            f"{len(missing)} utterances in {args.train_manifest} have no audio path"
        )
    from moonshine_voice.lora.data import load_wave as _load_wave

    for row in rows:
        if row.seconds is None:
            row.seconds = len(_load_wave(row.audio)) / SAMPLE_RATE
    eval_rows: List[Utterance] = []
    if args.eval_manifest:
        eval_rows = load_manifest(args.eval_manifest, args.data_root)
        for row in eval_rows:
            if row.seconds is None:
                row.seconds = len(_load_wave(row.audio)) / SAMPLE_RATE
    text_mode = choose_text_mode([r.text for r in rows], args.text_mode)
    print(
        f"manifest {args.train_manifest}: {len(rows)} utts / {hours_of(rows):.2f} h, "
        f"text mode '{text_mode}'"
    )
    train_hours = args.train_hours  # None = all except the dev slice

    def source(hours, pool=rows, mode=text_mode):
        return file_source(pool, hours, mode)

    return rows, eval_rows, Path(args.train_manifest).stem, source, text_mode, train_hours


def fit_adapter(
    model_id,
    processor,
    domain_index,
    domain_audio,
    *,
    replay_index=None,
    replay_audio=None,
    train_hours: Optional[float] = 2.0,
    dev_hours: float = 0.25,
    replay_dev_hours: float = 0.2,
    replay_ratio: float = 0.5,
    rank: int = 8,
    alpha=None,
    lr: float = 1e-3,
    batch_size: int = 8,
    max_steps: int = 3000,
    eval_every: int = 100,
    patience: int = 4,
    warmup: int = 100,
    seed: int = 0,
    device: str = "auto",
    tag: str = "",
    model=None,
    output_dir: Optional[Path] = None,
    adapter_path: Optional[Path] = None,
    extra_summary: Optional[dict] = None,
):
    """Train LoRA on already-built caches and return the merged model.

    ``replay_ratio=0`` skips replay mixing and general-domain stopping, even
    if a replay cache is passed. That is the notebook's no-replay ablation.
    """
    device = _device(device)
    amp_dtype = _amp_dtype(device)
    prefix = f"[{tag}] " if tag else ""

    if model is None:
        print(f"{prefix}loading {model_id}")
        model = MoonshineStreamingForConditionalGeneration.from_pretrained(
            model_id
        ).to(device)
    model.train()
    base_keys = set(model.state_dict())
    sites = add_lora(model, rank=rank, alpha=alpha, seed=seed)
    trainable = freeze_backbone(model, sites)
    total = sum(p.numel() for p in model.parameters())
    use_replay = replay_ratio > 0 and replay_index is not None
    print(
        f"{prefix}{trainable:,} trainable of {total:,} "
        f"({trainable / total * 100:.3f}%), replay {replay_ratio:.0%}"
    )

    entries = domain_index["entries"]
    dev, dev_secs = tail_split(entries, dev_hours)
    train, taken = [], 0.0
    train_cut = min(dev) if dev else len(entries)
    budget = (train_hours * 3600) if train_hours is not None else float("inf")
    for i in range(train_cut):
        if taken >= budget:
            break
        train.append(i)
        taken += entries[i]["samples"] / SAMPLE_RATE
    if train_hours is not None and taken < train_hours * 3600 * 0.95:
        print(
            f"{prefix}WARNING: only {taken / 3600:.2f} h available for a "
            f"{train_hours:g} h arm; cache more hours"
        )

    replay_entries = replay_index["entries"] if replay_index else []
    replay_dev, replay_secs, replay_pool = [], 0.0, []
    if use_replay:
        replay_dev, replay_secs = tail_split(replay_entries, replay_dev_hours)
        replay_pool = (
            list(range(min(replay_dev)))
            if replay_dev
            else list(range(len(replay_entries)))
        )
    print(
        f"{prefix}train {len(train)} utts / {taken / 3600:.2f} h | "
        f"dev {len(dev)} utts / {dev_secs / 3600:.2f} h"
        + (
            f" | replay pool {len(replay_pool)} utts, "
            f"dev {replay_secs / 3600:.2f} h"
            if use_replay
            else " | no replay"
        )
    )

    opt = torch.optim.AdamW(list(adapter_parameters(sites)), lr=lr, weight_decay=0.0)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    batch = batch_size

    def score_dev():
        def one(ents, blob, indices):
            model.eval()
            total_loss = count = 0
            with torch.no_grad():
                for group in make_batches(ents, indices, batch):
                    src, mask, dst = collate(ents, blob, group, device)
                    tokens = int((dst[:, 1:] != PAD).sum())
                    total_loss += (
                        batch_loss(model, src, mask, dst, amp_dtype).item() * tokens
                    )
                    count += tokens
            model.train()
            return total_loss / max(count, 1)

        in_domain = one(entries, domain_audio, dev) if dev else 0.0
        general = (
            one(replay_entries, replay_audio, replay_dev)
            if use_replay and replay_dev
            else None
        )
        return in_domain, general

    combine = lambda d, g: d if g is None else d + g
    first_dev, first_gen = score_dev()
    print(
        f"{prefix}before training (adapter is a no-op): in-domain {first_dev:.4f}"
        + (f"  general {first_gen:.4f}" if first_gen is not None else "")
    )
    best = {
        "score": combine(first_dev, first_gen),
        "dev": first_dev,
        "gen": first_gen,
        "step": 0,
        "state": adapter_state(sites),
    }
    rng = np.random.default_rng(seed)
    step, stale, stop, started = 0, 0, False, time.time()

    while step < max_steps and not stop:
        batches = [(False, b) for b in make_batches(entries, train, batch)]
        if use_replay:
            wanted = int(len(batches) * replay_ratio / max(1e-6, 1 - replay_ratio))
            pool = list(replay_pool)
            rng.shuffle(pool)
            replay_batches = make_batches(replay_entries, pool, batch)
            rng.shuffle(replay_batches)
            if replay_batches:
                batches += [
                    (True, replay_batches[i % len(replay_batches)])
                    for i in range(wanted)
                ]
        rng.shuffle(batches)
        for is_replay, group in batches:
            src, mask, dst = collate(
                *(
                    (replay_entries, replay_audio)
                    if is_replay
                    else (entries, domain_audio)
                ),
                group,
                device,
            )
            for pg in opt.param_groups:
                pg["lr"] = lr * min(1.0, (step + 1) / max(warmup, 1))
            loss = batch_loss(model, src, mask, dst, amp_dtype)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(adapter_parameters(sites)), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % eval_every == 0 or step == max_steps:
                d, g = score_dev()
                mark = ""
                if combine(d, g) < best["score"] - 1e-5:
                    best = {
                        "score": combine(d, g),
                        "dev": d,
                        "gen": g,
                        "step": step,
                        "state": adapter_state(sites),
                    }
                    stale, mark = 0, "  *best"
                else:
                    stale += 1
                shown = f"dev {d:.4f}"
                if g is not None:
                    shown += f"  general {g:.4f}"
                print(
                    f"  step {step:5d}  train {float(loss.detach()):.4f}  "
                    f"{shown}{mark}  ({step / (time.time() - started):.1f} steps/s)"
                )
                if stale >= patience:
                    print(f"  no improvement in {stale} evaluations; stopping")
                    stop = True
                    break
            if step >= max_steps:
                break

    print(
        f"{prefix}best step {best['step']}: in-domain {first_dev:.4f} -> {best['dev']:.4f}"
        + (
            f", general {first_gen:.4f} -> {best['gen']:.4f}"
            if first_gen is not None
            else ""
        )
    )
    load_adapter_state(sites, best["state"], device)
    merge_and_restore(model, sites)
    if set(model.state_dict()) != base_keys:
        raise SystemExit(
            "refusing to save: merged state dict does not match the base architecture"
        )

    adapter_file = Path(adapter_path) if adapter_path else None
    out = Path(output_dir) if output_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        if adapter_file is None:
            adapter_file = out / "adapter.safetensors"
    if adapter_file is not None:
        adapter_file.parent.mkdir(parents=True, exist_ok=True)
        save_file(best["state"], str(adapter_file))
    if out is not None:
        model.save_pretrained(out / "adapted")
        processor.save_pretrained(out / "adapted")
    if adapter_file is not None:
        size = adapter_file.stat().st_size / 1024
        extra = f" and {out / 'adapted'}" if out is not None else ""
        print(f"{prefix}wrote {adapter_file} ({size:.0f} KB){extra}")

    summary = {
        "model": model_id,
        "rank": rank,
        "lr": lr,
        "batch_size": batch_size,
        "trainable_parameters": trainable,
        "hours_actual": round(taken / 3600, 3),
        "baseline_dev_loss": first_dev,
        "best_dev_loss": best["dev"],
        "baseline_general_dev_loss": first_gen,
        "best_general_dev_loss": best["gen"],
        "best_step": best["step"],
        "steps_run": step,
        "replay_ratio": replay_ratio if use_replay else 0.0,
        "adapter_bytes": adapter_file.stat().st_size if adapter_file else 0,
    }
    if extra_summary:
        summary.update(extra_summary)
    if out is not None:
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return model


def train_adapter(
    args: Namespace,
    model=None,
    processor=None,
):
    """Run the recipe. ``args`` is the argparse namespace from ``__main__``."""
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_pool, eval_rows, domain, source, text_mode, train_hours = _load_train_rows(args)
    if text_mode == "lower":
        print("text mode 'lower' (corpus is uppercase; the model is not)")

    cache_hours = (train_hours or hours_of(train_pool)) + args.dev_hours
    if args.prepare_only:
        print(
            f"prepare-only: {domain} train pool {len(train_pool)} utts / "
            f"{hours_of(train_pool):.2f} h, will cache ~{cache_hours:.2f} h "
            f"in {work}"
        )
        if args.dataset == "atcosim":
            print(
                "ATCOSIM cannot be redistributed; get the corpus from TU Graz "
                "for anything beyond this Hub mirror. Split definition: "
                "moonshine-ai/atcosim-speaker-disjoint-splits"
            )
        return out

    device = _device(args.device)
    torch.manual_seed(args.seed)
    if device == "cpu":
        print("warning: training on CPU; a GPU is strongly recommended")

    if processor is None:
        processor = AutoProcessor.from_pretrained(args.model)

    def encode(text):
        return encode_text(processor, apply_text_mode(text, text_mode))

    domain_index = build_cache(
        domain, work, cache_hours, lambda h: source(h), encode
    )
    replay_index = None
    if not args.no_replay:
        replay_index = build_cache(
            "replay",
            work,
            args.replay_hours + args.replay_dev_hours,
            lambda h: replay_source(h, args.replay_repo),
            lambda text: encode_text(processor, text),
        )

    domain_audio = open_blob(work, domain, domain_index)
    replay_audio = (
        open_blob(work, "replay", replay_index) if replay_index else None
    )

    replay_ratio = 0.0 if args.no_replay else args.replay_ratio
    model = fit_adapter(
        args.model,
        processor,
        domain_index,
        domain_audio,
        replay_index=replay_index,
        replay_audio=replay_audio,
        train_hours=train_hours,
        dev_hours=args.dev_hours,
        replay_dev_hours=args.replay_dev_hours,
        replay_ratio=replay_ratio,
        rank=args.rank,
        alpha=args.alpha,
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        patience=args.patience,
        warmup=args.warmup,
        seed=args.seed,
        device=device,
        model=model,
        output_dir=out,
        extra_summary={
            "domain": domain,
            "model": args.model,
            "text_mode": text_mode,
        },
    )
    if args.eval:
        summary = json.loads((out / "summary.json").read_text())
        _run_eval(args, model, processor, device, eval_rows, domain, out, summary)
    return out


def _run_eval(args, model, processor, device, eval_rows, domain, out, summary):
    from moonshine_voice.lora.data import load_wave

    if eval_rows:
        idx = sample_indices(len(eval_rows), args.eval_limit, args.seed)
        chosen = [eval_rows[i] for i in idx]
        if args.dataset == "atcosim":
            waves = decode_atcosim(chosen)
        else:
            waves = [load_wave(r.audio) for r in chosen]
        refs = [r.text for r in chosen]
        print(f"scoring {len(chosen)} in-domain utterances")
        hyps = transcribe(model, processor, waves, device, batch_size=args.batch_size)
        summary["eval_wer"] = corpus_wer(refs, hyps)
        print(f"in-domain WER {summary['eval_wer']:.2f}%")
    if args.canary:
        print("scoring LibriSpeech test-clean canary")
        refs, waves = librispeech_eval(args.canary_limit, args.seed)
        hyps = transcribe(model, processor, waves, device, batch_size=args.batch_size)
        summary["canary_wer"] = corpus_wer(refs, hyps)
        print(f"LibriSpeech WER {summary['canary_wer']:.2f}%")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
