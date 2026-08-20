"""Rank-r LoRA on self-attention q/k/v, with a shared down-projection.

Imported only after ``require_lora_deps()``: this module loads PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

QKV = ("q_proj", "k_proj", "v_proj")
VALID_SITES = ("decoder", "encoder", "both")
VALID_ADAPT = ("lora", "full")


class LoRALinear(nn.Module):
    """A frozen linear with a rank-r side path.

    ``down`` is passed in so q/k/v of one layer can share a single A. ``up``
    starts at zero, so an untrained adapter is exactly the base model.
    """

    def __init__(self, base, down, rank, scale=1.0):
        super().__init__()
        self.base, self.down = base, down
        self.up = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        nn.init.zeros_(self.up.weight)
        self.scale, self.merged = scale, False

    def delta(self):
        return self.scale * (self.up.weight @ self.down.weight)

    def forward(self, x):
        out = self.base(x)
        return out if self.merged else out + self.scale * self.up(self.down(x))

    @torch.no_grad()
    def merge_(self):
        if not self.merged:
            self.base.weight.add_(self.delta())
            self.merged = True


def _wrap_stack(layers, prefix, rank, scale, gen, sites):
    """Replace each layer's self-attention q/k/v. Keys are ``{prefix}.{i}``."""
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        bases = [getattr(attn, role) for role in QKV]
        down = nn.Linear(
            bases[0].in_features,
            rank,
            bias=False,
            device=bases[0].weight.device,
            dtype=bases[0].weight.dtype,
        )
        weight = torch.empty(rank, bases[0].in_features)
        nn.init.kaiming_uniform_(weight, a=5**0.5, generator=gen)
        with torch.no_grad():
            down.weight.copy_(weight.to(down.weight.device, down.weight.dtype))
        group = []
        for role, base in zip(QKV, bases):
            wrapped = LoRALinear(base, down, rank, scale)
            setattr(attn, role, wrapped)
            group.append(wrapped)
        sites[f"{prefix}.{i}"] = group


def add_lora(model, rank=8, alpha=None, seed=0, sites="decoder"):
    """Replace self-attention q/k/v with a ``LoRALinear`` on the chosen stacks.

    ``sites`` is ``decoder`` (default), ``encoder``, or ``both``. Cross-attention
    is never wrapped: those projections run over the whole audio memory span
    and are the expensive site.
    """
    if sites not in VALID_SITES:
        raise ValueError(f"sites must be one of {VALID_SITES}, got {sites!r}")
    alpha = float(rank if alpha is None else alpha)
    scale = alpha / rank
    gen = torch.Generator(device="cpu").manual_seed(seed)
    wrapped = {}
    if sites in ("decoder", "both"):
        _wrap_stack(model.model.decoder.layers, "decoder", rank, scale, gen, wrapped)
    if sites in ("encoder", "both"):
        _wrap_stack(model.model.encoder.layers, "encoder", rank, scale, gen, wrapped)
    return wrapped


def adapter_parameters(sites):
    """Yield each adapter tensor once (shared A is not counted three times)."""
    seen = set()
    for group in sites.values():
        for module in group:
            for param in (module.down.weight, module.up.weight):
                if id(param) not in seen:
                    seen.add(id(param))
                    yield param


def freeze_backbone(model, sites):
    for param in model.parameters():
        param.requires_grad_(False)
    for param in adapter_parameters(sites):
        param.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_adaptation(model, adapt="lora", sites="decoder", rank=8, alpha=None, seed=0):
    """Either attach LoRA or unfreeze the whole backbone.

    Returns ``(lora_sites, trainable_count)``. ``lora_sites`` is ``None`` for
    a full fine-tune, so callers skip adapter save / merge.
    """
    if adapt not in VALID_ADAPT:
        raise ValueError(f"adapt must be one of {VALID_ADAPT}, got {adapt!r}")
    if adapt == "full":
        for param in model.parameters():
            param.requires_grad_(True)
        return None, sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_sites = add_lora(model, rank=rank, alpha=alpha, seed=seed, sites=sites)
    return lora_sites, freeze_backbone(model, lora_sites)


def adapter_state(sites):
    state = {}
    for key, group in sites.items():
        state[f"{key}.down"] = group[0].down.weight.detach().cpu().clone()
        for role, module in zip(QKV, group):
            state[f"{key}.{role}.up"] = module.up.weight.detach().cpu().clone()
    return state


def load_adapter_state(sites, state, device):
    with torch.no_grad():
        for key, group in sites.items():
            group[0].down.weight.copy_(state[f"{key}.down"].to(device))
            for role, module in zip(QKV, group):
                module.up.weight.copy_(state[f"{key}.{role}.up"].to(device))


def _stack_layers(model, prefix):
    return getattr(model.model, prefix).layers


def merge_and_restore(model, sites):
    """Fold the side paths in and put the plain ``nn.Linear`` back.

    Skip this and ``save_pretrained`` sees ``q_proj.base.weight`` keys plus
    three tensors sharing one ``down.weight``, which safetensors rejects.
    """
    for key, group in sites.items():
        prefix, index = key.split(".")
        attn = _stack_layers(model, prefix)[int(index)].self_attn
        for role, module in zip(QKV, group):
            module.merge_()
            setattr(attn, role, module.base)
