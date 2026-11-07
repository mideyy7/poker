"""
Strength and robustness tests for our bot (not submitted).

Fast by default. The full gate (candidate vs frozen baseline over the
whole gauntlet) is slow, so it only runs with STRENGTH=1:

    STRENGTH=1 .venv/Scripts/python -m pytest tests/test_strength.py -q

The item tests check each strong.md mechanism does what it claims;
whether it makes the bot stronger is decided by the gate.
"""

import os
import random

import pytest

from arena.dummies import CallBot, RandomBot, ShoveBot
from card_utils import full_deck
from engine import play_hand
from model import Model
from tests.gate import Timed, duplicate, gate, mean_ci

SLOW = pytest.mark.skipif(not os.environ.get("STRENGTH"),
                          reason="slow: set STRENGTH=1")
DEALS = int(os.environ.get("STRENGTH_DEALS", "400"))


class WeakModel(Model):
    """A deliberately worse bot: never bets or raises after the flop.
    The gate must catch it."""

    def _postflop(self, gs, me, opp, team, board, combos):
        if gs.call_amount == 0:
            return self._call()
        rng = self._rng(gs)
        eq = self._eq(board, combos, rng)
        if eq > gs.pot_odds:
            return self._call()
        return self._fold_or_check(gs)


# -- the gate itself ----------------------------------------------------

def test_gate_identical_bots_differ_by_exactly_zero(capsys):
    assert gate("model:Model", "model:Model", ["call", "random"], deals=20,
                h2h_deals=0, seed=3, samples=150, workers=2)
    assert "+0.0 +/-    0.0" in capsys.readouterr().out


def test_gate_catches_a_weaker_bot():
    ok = gate("tests.test_strength:WeakModel", "model:Model",
              ["call", "limpcall"], deals=150, h2h_deals=0, seed=1,
              samples=200, workers=2)
    assert not ok


@SLOW
def test_candidate_passes_full_gate():
    from tests.gate import GAUNTLET
    assert gate("model:Model", "tests.baseline.bl_model:Model", GAUNTLET,
                deals=DEALS, h2h_deals=DEALS, seed=11, samples=400,
                workers=4)


# -- exploitation targets ---------------------------------------------

def _bb_call_rate_vs_jam(hands=300):
    """Share of hands where we, in the BB, call ShoveBot's open jam."""
    me = Model(samples=150)
    calls = faced = 0
    for i in range(hands):
        deck = full_deck()
        random.Random(i).shuffle(deck)
        r = play_hand([ShoveBot(), me], deck, [500, 500], button=0,
                      teams=["shover", "me"])
        moves = [a for a in r.actions if a.player_id == 1]
        if moves:
            faced += 1
            calls += moves[0].kind == "call"
    return calls / faced


def test_calls_a_100_percent_jammer_wide():
    # Vs a random-hand jam the break-even call needs ~48% equity: about
    # 60% of hands qualify. Allow for the learning period.
    assert _bb_call_rate_vs_jam() > 0.5


def test_value_bets_a_calling_station():
    me = Timed(Model(samples=200))
    nets, _ = duplicate(me, CallBot, 150, seed=5)
    m, h = mean_ci(nets)
    assert m - h > 150  # bb/100: big edge, not just "ahead"


# -- the strong.md mechanisms -------------------------------------------
# (Whether they make the bot STRONGER is decided by the gate; these check
# that each mechanism does what it claims.)

def test_item1_decay_tracks_a_change_of_style():
    from pk_opp import Stats
    fast, slow = Stats(decay=0.985), Stats(decay=1.0)
    for st in (fast, slow):
        for _ in range(150):
            st.add("t", "sb_open", "jam")
        for _ in range(100):
            st.add("t", "sb_open", "raise")
    assert fast.freq("t", "sb_open", "raise") > 0.6
    assert slow.freq("t", "sb_open", "raise") < 0.45


def test_item2_learns_bluffs_from_showdowns():
    from tests.bench import RiverBluffer
    m = Model(samples=150)
    for i in range(200):
        deck = full_deck()
        random.Random(i).shuffle(deck)
        play_hand([m, RiverBluffer()], deck, [500, 500], button=i % 2,
                  teams=["me", "bluffer"])
    assert m.stats.n("bluffer", "sd_bigbet") >= 8
    assert m.stats.freq("bluffer", "sd_bigbet", "weak") > 0.35


def test_item2_bluff_floor_rises_with_bluff_share():
    from pk_opp import Event
    m = Model()
    e = Event(1, "river", "post_first", "bet", 300, 5, 200, 0)
    low = m._bluff_floor("t", 0.4, e)
    for _ in range(50):
        m.stats.add("t", "sd_bigbet", "weak")
    assert m._bluff_floor("t", 0.4, e) > low
    small = Event(1, "river", "post_first", "bet", 50, 5, 200, 0)
    assert m._bluff_floor("t", 0.4, small) is None  # small bets: unchanged


def test_item3_size_buckets():
    from pk_opp import size_bucket
    assert size_bucket(30, 100) == "s"
    assert size_bucket(60, 100) == "m"
    assert size_bucket(100, 100) == "l"


def test_item6_loss_limit_trusts_prior_more_when_losing():
    class GS:
        stacks = (300, 700)
    m = Model()
    m.player_id = 0
    for _ in range(149):
        m._track_result("t", GS)
    assert m.stats.weight.get("t", 1.0) == 1.0  # not before 150 hands
    m._track_result("t", GS)
    assert m.stats.weight["t"] == 4.0


def test_features_off_is_exactly_v1():
    """With every feature off the bot must play exactly like the frozen
    baseline (paired difference identically zero)."""
    from tests.baseline.bl_model import Model as Base
    from tests.variants import NoFeatures
    a, _ = duplicate(Timed(NoFeatures(samples=150)), CallBot, 25, seed=9)
    b, _ = duplicate(Timed(Base(samples=150)), CallBot, 25, seed=9)
    assert a == b


# -- robustness -------------------------------------------------------

@pytest.mark.parametrize("stacks", [[500, 500], [500, 130], [37, 500],
                                    [1000, 400], [21, 25]])
def test_odd_stacks_never_fault(stacks):
    me = Timed(Model(samples=150))
    for i in range(60):
        deck = full_deck()
        random.Random(i).shuffle(deck)
        for seat in (0, 1):
            opp = RandomBot(seed=i)
            models = [me, opp] if seat == 0 else [opp, me]
            s = stacks if seat == 0 else stacks[::-1]
            r = play_hand(models, deck, s, button=i % 2,
                          teams=["me", "r"] if seat == 0 else ["r", "me"])
            assert seat not in r.faults, r.faults
    assert me.model.errors == 0
    assert me.slowest < 1.5


def test_three_handed_table_never_faults():
    me = Timed(Model(samples=150))
    for i in range(60):
        deck = full_deck()
        random.Random(i).shuffle(deck)
        r = play_hand([me, RandomBot(seed=i), CallBot()], deck,
                      [500, 500, 500], button=i % 3, teams=["me", "r", "c"])
        assert 0 not in r.faults, r.faults
    assert me.model.errors == 0
