"""Our bot's own tests (not submitted)."""

import pickle
import random

import pytest

from arena.dummies import CallBot, RandomBot, ShoveBot
from card_utils import Card, evaluate, full_deck
from engine import play_hand
from model import Model
from pk_equity import equity
from pk_eval import cards_to_ints, evaluate_ints, parse_ints
from pk_opp import Stats
from pk_preflop import EQ
from pk_ranges import INDEX, VS_RANDOM, full_range, preflop_equity, to_combos


def _sign(x):
    return (x > 0) - (x < 0)


# -- evaluator ------------------------------------------------------------

def test_evaluator_matches_official():
    deck = full_deck()
    rng = random.Random(7)
    for _ in range(5000):
        n = rng.choice([5, 6, 7])
        a, b = rng.sample(deck, n), rng.sample(deck, n)
        fast = evaluate_ints(cards_to_ints(a)) - evaluate_ints(cards_to_ints(b))
        slow = (evaluate(a) > evaluate(b)) - (evaluate(a) < evaluate(b))
        assert _sign(fast) == slow


@pytest.mark.parametrize("better,worse", [
    ("AS 2D 3C 4H 5S 9D KC", "AS KD QC 9H 7S 3D 2C"),  # wheel > high card
    ("6S 2D 3C 4H 5S 9D KC", "AS 2D 3C 4H 5S 9D KC"),  # 6-high > wheel
    ("AH 2H 3H 4H 5H KS KD", "KH QH JH 9H 7H KS KD"),  # steel wheel > flush
    ("9S 9D 9C 9H AS 2D 3C", "9S 9D 9C 9H KS 2D 3C"),  # quads kicker
    ("AS AD KC KH 2S 2D 7C", "AS AD QC QH 3S 3D KC"),  # best two pairs
    ("KS KD KC 2H 2S QD QC", "KS KD KC 2H 2S JD JC"),  # boat picks QQ
])
def test_categories(better, worse):
    assert evaluate_ints(parse_ints(better)) > evaluate_ints(parse_ints(worse))
    official = [Card.parse(t) for t in better.split()]
    other = [Card.parse(t) for t in worse.split()]
    assert evaluate(official) > evaluate(other)


# -- equity ---------------------------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    ("AA", "KK", 0.82), ("AKs", "QQ", 0.46), ("72o", "AA", 0.12),
])
def test_matrix_known_equities(a, b, expected):
    assert abs(EQ[INDEX[a]][INDEX[b]] - expected) < 0.025


def test_vs_random():
    assert abs(VS_RANDOM[INDEX["AA"]] - 0.85) < 0.015
    assert abs(VS_RANDOM[INDEX["72o"]] - 0.35) < 0.02


def test_monte_carlo_equity_vs_specific_hand():
    hero = parse_ints("AS AD")
    villain = [(a, b, 1.0) for a, b in [tuple(parse_ints("KS KD"))]]
    eq = equity(hero, [], villain, 20000, random.Random(1))
    assert abs(eq - 0.82) < 0.015


def test_river_equity_exact():
    hero = parse_ints("AS AD")
    board = parse_ints("AH AC 2D 7S 9C")  # quad aces: unbeatable
    combos = to_combos(full_range(), set(hero) | set(board))
    assert equity(hero, board, combos, 100, random.Random(1)) == 1.0


def test_preflop_equity_card_removal_runs():
    hero = parse_ints("AS KS")
    assert 0.6 < preflop_equity(hero, full_range()) < 0.7


# -- play -----------------------------------------------------------------

def _rigged(hero, villain, board):
    cards = [Card.parse(t) for t in (hero + " " + villain + " " + board).split()]
    rest = [c for c in full_deck() if c not in cards]
    return cards + rest


def _hero_moves(result, seat):
    return [a for a in result.actions if a.player_id == seat]


@pytest.mark.parametrize("button", [0, 1])
def test_aces_never_fold_vs_shove(button):
    deck = _rigged("AS AD", "7C 2H", "3D 8S 9H KC 4D")
    r = play_hand([Model(), ShoveBot()], deck, [500, 500], button=button)
    assert not r.faults
    assert all(a.kind != "fold" for a in _hero_moves(r, 0))
    assert r.net[0] > 0


@pytest.mark.parametrize("button", [0, 1])
def test_nuts_never_fold(button):
    deck = _rigged("AH KH", "2C 2D", "QH JH TH 2S 7D")
    r = play_hand([Model(), ShoveBot()], deck, [500, 500], button=button)
    assert all(a.kind != "fold" for a in _hero_moves(r, 0))
    assert r.net[0] > 0


def test_never_fold_when_free_and_no_errors():
    folded_free = []

    class Watch(Model):
        def run(self, actions, gs):
            action = super().run(actions, gs)
            if action.kind == "fold" and gs.call_amount == 0:
                folded_free.append(action)
            return action

    me = Watch(samples=200)
    for i in range(150):
        deck = full_deck()
        random.Random(i).shuffle(deck)
        for opp in (RandomBot(seed=i), CallBot()):
            r = play_hand([me, opp], deck, [500, 500], button=i % 2,
                          teams=["me", type(opp).__name__])
            assert 0 not in r.faults, r.faults
    assert not folded_free
    assert me.errors == 0


def test_stats_gating():
    s = Stats()
    for _ in range(5):
        s.add("t", "post_facing", "fold")
    assert s.freq("t", "post_facing", "fold") < 0.7  # still near the prior
    for _ in range(300):
        s.add("t", "post_facing", "fold")
    assert s.freq("t", "post_facing", "fold") > 0.9
    assert s.freq("other", "post_facing", "fold") == pytest.approx(0.40)


def test_learns_shover_and_calls_wider():
    m = Model(samples=200)
    for i in range(100):
        deck = full_deck()
        random.Random(i).shuffle(deck)
        play_hand([m, ShoveBot()], deck, [500, 500], button=i % 2,
                  teams=["me", "shover"])
    assert m.stats.freq("shover", "sb_open", "jam") > 0.8


def test_pickle_round_trip_after_play():
    m = Model(samples=200)
    for i in range(30):
        deck = full_deck()
        random.Random(i).shuffle(deck)
        play_hand([m, RandomBot(seed=i)], deck, [500, 500], button=i % 2,
                  teams=["me", "rnd"])
    data = pickle.dumps(m)
    assert len(data) < 5 << 20
    m2 = pickle.loads(data)
    assert m2.stats.counts == m.stats.counts
