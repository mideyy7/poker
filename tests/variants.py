"""
Bot variants for gating one strong.md item at a time:

    python -m tests.gate --candidate tests.variants:Only_decay
    python -m tests.gate --candidate tests.variants:NoFeatures  # == v1
"""

from model import ALL_FEATURES, Model


def _variant(features):
    class Variant(Model):
        def __init__(self, samples=1200):
            super().__init__(samples=samples, features=features)
    return Variant


NoFeatures = _variant(())
for _f in ALL_FEATURES:
    globals()["Only_" + _f] = _variant((_f,))
    globals()["All_but_" + _f] = _variant(
        tuple(f for f in ALL_FEATURES if f != _f))
