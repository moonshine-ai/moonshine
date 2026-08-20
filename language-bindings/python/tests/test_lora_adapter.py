"""Adapter invariants. Skipped unless the ``[lora]`` extra is installed."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from moonshine_voice.lora.adapter import (  # noqa: E402
    LoRALinear,
    adapter_parameters,
    add_lora,
    freeze_backbone,
    merge_and_restore,
    prepare_adaptation,
)


class _Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)


class _Layer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = _Attn(dim)


class _Decoder(nn.Module):
    def __init__(self, depth, dim):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(dim) for _ in range(depth)])


class _Encoder(_Decoder):
    pass


class _Dummy(nn.Module):
    """Enough of MoonshineStreamingForConditionalGeneration for add_lora."""

    def __init__(self, depth=2, dim=32, encoder=False):
        super().__init__()
        self.model = nn.Module()
        self.model.decoder = _Decoder(depth, dim)
        if encoder:
            self.model.encoder = _Encoder(depth, dim)


def test_untrained_adapter_is_a_noop():
    torch.manual_seed(0)
    model = _Dummy()
    x = torch.randn(2, 4, 32)
    before = model.model.decoder.layers[0].self_attn.q_proj(x)
    add_lora(model, rank=4, seed=0)
    after = model.model.decoder.layers[0].self_attn.q_proj(x)
    assert torch.equal(before, after)
    wrapped = model.model.decoder.layers[0].self_attn.q_proj
    assert isinstance(wrapped, LoRALinear)
    assert wrapped.up.weight.count_nonzero() == 0


def test_only_adapter_trains():
    model = _Dummy()
    sites = add_lora(model, rank=4, seed=1)
    trainable = freeze_backbone(model, sites)
    expected = sum(p.numel() for p in adapter_parameters(sites))
    assert trainable == expected
    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert frozen, "backbone should be frozen"
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in frozen)


def test_merge_restores_plain_linears():
    model = _Dummy()
    base_keys = set(model.state_dict())
    sites = add_lora(model, rank=4, seed=2)
    merge_and_restore(model, sites)
    assert set(model.state_dict()) == base_keys
    assert not isinstance(model.model.decoder.layers[0].self_attn.q_proj, LoRALinear)


def test_encoder_and_both_sites():
    model = _Dummy(encoder=True)
    sites = add_lora(model, rank=4, seed=3, sites="encoder")
    assert all(key.startswith("encoder.") for key in sites)
    assert isinstance(model.model.encoder.layers[0].self_attn.q_proj, LoRALinear)
    assert not isinstance(model.model.decoder.layers[0].self_attn.q_proj, LoRALinear)

    model = _Dummy(encoder=True)
    sites = add_lora(model, rank=4, seed=4, sites="both")
    assert any(key.startswith("encoder.") for key in sites)
    assert any(key.startswith("decoder.") for key in sites)
    merge_and_restore(model, sites)
    assert not isinstance(model.model.encoder.layers[0].self_attn.q_proj, LoRALinear)
    assert not isinstance(model.model.decoder.layers[0].self_attn.q_proj, LoRALinear)


def test_full_adapt_leaves_plain_linears():
    model = _Dummy(encoder=True)
    base_keys = set(model.state_dict())
    sites, trainable = prepare_adaptation(model, adapt="full")
    assert sites is None
    assert trainable == sum(p.numel() for p in model.parameters())
    assert set(model.state_dict()) == base_keys
    assert not isinstance(model.model.decoder.layers[0].self_attn.q_proj, LoRALinear)
    assert all(p.requires_grad for p in model.parameters())


def test_default_lr_depends_on_adapt_and_sites():
    pytest.importorskip("transformers")
    from moonshine_voice.lora.train import default_lr

    assert default_lr("lora", "decoder") == 1e-3
    assert default_lr("lora", "both") == 1e-4
    assert default_lr("lora", "encoder") == 1e-4
    assert default_lr("full", "decoder") == 1e-5


def test_sample_indices_and_tail_split():
    pytest.importorskip("transformers")
    from moonshine_voice.lora.train import sample_indices, tail_split

    assert sample_indices(4, None, 0) == [0, 1, 2, 3]
    chosen = sample_indices(10, 3, 0)
    assert len(chosen) == 3
    assert chosen == sorted(chosen)
    entries = [{"samples": 16_000} for _ in range(10)]
    held, taken = tail_split(entries, 3 / 3600)
    assert len(held) == 3
    assert abs(taken - 3.0) < 1e-6
