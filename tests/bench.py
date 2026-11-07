"""
Sparring bots that imitate likely student entries, and a fast in-process
duplicate harness (no worker processes, no time limit checks).

    .venv/Scripts/python -m tests.bench --deals 300 [--samples 400]
"""

import argparse
import math
import random
import sys
import time

from arena.dummies import DUMMIES, _Bot
from card_utils import evaluate, full_deck
from engine import play_hand
from model import Model
from player_action import PlayerAction


def _strength(hand):
    lo, hi = sorted(c.value for c in hand)
    score = hi * 2 + lo + (10 if hi == lo else 0) + (2 if hand[0].suit == hand[1].suit else 0)
    return score  # roughly 7..48


class MinRaiser(_Bot):
    """Min-raises every hand preflop, c-bets half pot, calls bets < pot."""

    def run(self, actions, gs):
        if gs.street == "preflop":
            if gs.can_raise and gs.bets[self.player_id] < 3 * gs.big_blind:
                return self._raise_to(gs, gs.min_raise_to)
            return PlayerAction.call(self.player_id)
        if gs.call_amount == 0:
            return self._raise_to(gs, gs.pot // 2)
        if gs.call_amount < gs.pot:
            return PlayerAction.call(self.player_id)
        return PlayerAction.fold(self.player_id)


class LimpCaller(_Bot):
    """Never raises; calls anything up to the pot size."""

    def run(self, actions, gs):
        if gs.call_amount <= gs.pot:
            return PlayerAction.call(self.player_id)
        return PlayerAction.fold(self.player_id)


class Nit(_Bot):
    """Jams the top ~15% preflop, otherwise check/folds; calls postflop."""

    def run(self, actions, gs):
        if gs.street == "preflop":
            if _strength(self.hand) >= 36:
                return self._raise_to(gs, gs.max_raise_to)
            return self._check_or_fold(gs)
        return PlayerAction.call(self.player_id)


class Aggro(_Bot):
    """Raises 3x with ~60% preflop, bets 75% pot with anything, keeps
    betting with a pair or better, folds to raises without one."""

    def run(self, actions, gs):
        if gs.street == "preflop":
            if _strength(self.hand) >= 22:
                if gs.can_raise and gs.bets[self.player_id] < 4 * gs.big_blind:
                    return self._raise_to(gs, 3 * gs.current_bet)
                return PlayerAction.call(self.player_id)
            return self._check_or_fold(gs)
        made = evaluate(self.hand + list(gs.board))[0] >= 1
        if gs.call_amount == 0:
            return self._raise_to(gs, int(gs.pot * 0.75))
        if made:
            return PlayerAction.call(self.player_id)
        return PlayerAction.fold(self.player_id)


SPARRING = {
    "minraise": MinRaiser,
    "limpcall": LimpCaller,
    "nit": Nit,
    "aggro": Aggro,
}


def duplicate(make_me, make_opp, deals, seed=0):
    """bb/100 for me, its 95% half-width, and my error count."""
    me, opp = make_me(), make_opp()
    nets = []
    for i in range(deals):
        deck = full_deck()
        random.Random(seed * 100003 + i).shuffle(deck)
        pair = 0
        for seat in (0, 1):
            models = [me, opp] if seat == 0 else [opp, me]
            teams = ["me", "opp"] if seat == 0 else ["opp", "me"]
            r = play_hand(models, deck, [500, 500], button=0, teams=teams)
            if seat in r.faults:
                print("FAULT", r.faults[seat], file=sys.stderr)
            pair += r.net[seat]
        nets.append(pair / 2 / 20)  # bb per hand, averaged over the pair
    mean = sum(nets) / len(nets)
    var = sum((x - mean) ** 2 for x in nets) / max(1, len(nets) - 1)
    half = 1.96 * math.sqrt(var / len(nets))
    return mean * 100, half * 100, getattr(me, "errors", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--opponents", nargs="+",
                    default=list(SPARRING) + ["fold", "call", "random",
                                               "shove", "tight", "self"])
    args = ap.parse_args()
    for name in args.opponents:
        if name == "self":
            make_opp = lambda: Model(samples=args.samples)  # noqa: E731
        elif name in SPARRING:
            make_opp = SPARRING[name]
        else:
            make_opp = lambda n=name: DUMMIES[n](seed=1)  # noqa: E731
        t = time.time()
        bb100, half, errors = duplicate(
            lambda: Model(samples=args.samples), make_opp, args.deals)
        print(f"{name:<10} {bb100:+8.1f} +/- {half:6.1f} bb/100  "
              f"errors={errors}  {time.time() - t:5.1f}s", flush=True)


if __name__ == "__main__":
    main()
