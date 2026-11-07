"""
Strength gate: is the candidate bot at least as strong as the baseline?

    .venv/Scripts/python -m tests.gate                 # default gauntlet
    .venv/Scripts/python -m tests.gate --deals 600 --h2h 800

Both bots play the SAME decks against fresh copies of the SAME opponents
(duplicate: every deck twice, seats swapped). Per opponent it reports the
candidate's bb/100 and the paired difference candidate - baseline with a
95% interval, then a verdict:

  FAIL  a fault, an exception fallback, a move over the time limit, a
        sanity bot the candidate does not clearly beat, any opponent
        where the candidate is significantly WORSE than the baseline, or
        the candidate significantly losing head-to-head to the baseline.
  PASS  none of those, and the average paired difference is >= 0.

A change ships only on PASS. Exit code 0 on PASS, 1 on FAIL.
"""

import argparse
import importlib
import math
import random
import sys
import time
from multiprocessing import Pool

from arena.dummies import DUMMIES
from card_utils import full_deck
from engine import play_hand
from tests.bench import SPARRING

SANITY = ["fold", "call", "random", "shove"]
GAUNTLET = SANITY + ["tight", "equity"] + list(SPARRING)
TIME_LIMIT = 1.5  # seconds per move; the real limit is 10


def load(spec):
    module, _, cls = spec.partition(":")
    return getattr(importlib.import_module(module), cls or "Model")


def make_opponent(name, seed):
    if name in SPARRING:
        return SPARRING[name](seed=seed)
    return DUMMIES[name](seed=seed)


class Timed:
    """Wraps a model: records the slowest move and keeps its state."""

    def __init__(self, model):
        self.model = model
        self.slowest = 0.0

    def seed(self, hand, player_id):
        self.model.seed(hand, player_id)

    def run(self, actions, gs):
        t = time.perf_counter()
        action = self.model.run(actions, gs)
        self.slowest = max(self.slowest, time.perf_counter() - t)
        return action

    def end_game(self, actions, gs):
        self.model.end_game(actions, gs)


def duplicate(me, make_opp, deals, seed):
    """Per-deal bb won by `me` (averaged over both seats) and faults."""
    opp = make_opp()
    nets, faults = [], 0
    for i in range(deals):
        deck = full_deck()
        random.Random(seed * 1_000_003 + i).shuffle(deck)
        pair = 0
        for seat in (0, 1):
            models = [me, opp] if seat == 0 else [opp, me]
            teams = ["me", "opp"] if seat == 0 else ["opp", "me"]
            r = play_hand(models, deck, [500, 500], button=0, teams=teams)
            pair += r.net[seat]
            faults += seat in r.faults
        nets.append(pair / 2 / 20)
    return nets, faults


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / max(1, n - 1)
    return m * 100, 1.96 * math.sqrt(v / n) * 100  # bb/100


def _job(args):
    """One opponent: candidate and baseline on identical decks."""
    cand_spec, base_spec, opp_name, deals, seed, samples = args
    out = {"opp": opp_name}
    for tag, spec in (("cand", cand_spec), ("base", base_spec)):
        model = Timed(load(spec)(samples=samples))
        nets, faults = duplicate(
            model, lambda: make_opponent(opp_name, seed), deals, seed)
        out[tag] = nets
        out[tag + "_faults"] = faults
        out[tag + "_errors"] = getattr(model.model, "errors", 0)
        out[tag + "_slowest"] = model.slowest
    return out


def _h2h(args):
    cand_spec, base_spec, deals, seed, samples = args
    me = Timed(load(cand_spec)(samples=samples))
    nets, faults = duplicate(
        me, lambda: load(base_spec)(samples=samples), deals, seed)
    return nets, faults, getattr(me.model, "errors", 0), me.slowest


def gate(cand, base, opponents, deals, h2h_deals, seed, samples, workers):
    jobs = [(cand, base, o, deals, seed + k, samples)
            for k, o in enumerate(opponents)]
    failures = []
    diffs = []
    print(f"{'opponent':<11} {'candidate bb/100':>20} {'cand - base':>20}"
          f"  flags")
    with Pool(workers) as pool:
        h2h = pool.apply_async(_h2h, ((cand, base, h2h_deals, seed + 999,
                                       samples),)) if h2h_deals else None
        for r in pool.imap(_job, jobs):
            c_m, c_h = mean_ci(r["cand"])
            d = [a - b for a, b in zip(r["cand"], r["base"])]
            d_m, d_h = mean_ci(d)
            diffs.append(d_m)
            flags = []
            if r["cand_faults"]:
                flags.append(f"{r['cand_faults']} faults")
            if r["cand_errors"]:
                flags.append(f"{r['cand_errors']} errors")
            if r["cand_slowest"] > TIME_LIMIT:
                flags.append(f"slow {r['cand_slowest']:.2f}s")
            if r["opp"] in SANITY and c_m - c_h <= 0:
                flags.append("does not clearly beat a sanity bot")
            if d_m + d_h < 0:
                flags.append("significantly WORSE than baseline")
            print(f"{r['opp']:<11} {c_m:+9.1f} +/- {c_h:6.1f}   "
                  f"{d_m:+9.1f} +/- {d_h:6.1f}  {'; '.join(flags)}",
                  flush=True)
            failures += [f"{r['opp']}: {f}" for f in flags]
        if h2h:
            nets, faults, errors, slowest = h2h.get()
            m, h = mean_ci(nets)
            print(f"{'vs baseline':<11} {m:+9.1f} +/- {h:6.1f}  "
                  f"(head-to-head, {h2h_deals} deals)")
            if m + h < 0:
                failures.append("loses head-to-head to the baseline")
            if faults or errors or slowest > TIME_LIMIT:
                failures.append("head-to-head: fault/error/slow")
    avg = sum(diffs) / len(diffs)
    print(f"\naverage paired difference: {avg:+.1f} bb/100")
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return False
    if avg < 0:
        print("FAIL\n  - average paired difference is negative")
        return False
    print("PASS")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tests.gate")
    ap.add_argument("--candidate", default="model:Model")
    ap.add_argument("--baseline", default="tests.baseline.bl_model:Model")
    ap.add_argument("--opponents", nargs="+", default=GAUNTLET)
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--h2h", type=int, default=600,
                    help="head-to-head deals vs the baseline (0 = skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=400,
                    help="Monte Carlo samples for both bots (speed)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    ok = gate(args.candidate, args.baseline, args.opponents, args.deals,
              args.h2h, args.seed, args.samples, args.workers)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
