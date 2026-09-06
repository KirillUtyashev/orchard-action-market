"""Verify that set_all_seeds() actually controls rng references held by other modules."""

from orchard.seed import set_all_seeds
from orchard.seed import rng as seed_rng
from orchard.trainer.value_base import rng as value_rng
from orchard.env.stochastic import rng as stochastic_rng


def test_all_modules_share_same_rng_instance():
    assert seed_rng is value_rng, "value_base.rng is not the same object as seed.rng"
    assert seed_rng is stochastic_rng, "stochastic.rng is not the same object as seed.rng"


def test_seed_produces_identical_sequences_across_calls():
    set_all_seeds(1234)
    seq1 = [seed_rng.random() for _ in range(10)]
    vseq1 = [value_rng.random() for _ in range(10)]
    sseq1 = [stochastic_rng.random() for _ in range(10)]

    set_all_seeds(1234)
    seq2 = [seed_rng.random() for _ in range(10)]
    vseq2 = [value_rng.random() for _ in range(10)]
    sseq2 = [stochastic_rng.random() for _ in range(10)]

    assert seq1 == seq2, "seed.rng sequence not reproducible"
    assert vseq1 == vseq2, "value_base.rng sequence not reproducible"
    assert sseq1 == sseq2, "stochastic.rng sequence not reproducible"


def test_cross_module_sequences_are_interleaved_correctly():
    """Drawing from any module advances the shared RNG state."""
    set_all_seeds(42)
    v1 = value_rng.random()
    s1 = stochastic_rng.random()

    set_all_seeds(42)
    r1 = seed_rng.random()
    r2 = seed_rng.random()

    assert v1 == r1, "first draw from value_rng should match first draw from seed_rng"
    assert s1 == r2, "first draw from stochastic_rng should match second draw from seed_rng"
