"""
Equity of a hero hand against a weighted villain range.

A range here is a list of (card_a, card_b, weight) with int cards
(see pk_eval). Combos that clash with the hero hand or board are skipped.
"""

import time

from pk_eval import evaluate_ints


def live_combos(combos, dead):
    """Drop combos that use a dead card or have no weight."""
    return [c for c in combos
            if c[2] > 0 and c[0] not in dead and c[1] not in dead]


def equity(hero, board, combos, samples, rng, deadline=None):
    """Share of the pot hero wins (ties split) vs the weighted range.

    Exact on the river; Monte Carlo before it. deadline is a
    time.perf_counter() value after which sampling stops early.
    """
    dead = set(hero) | set(board)
    combos = live_combos(combos, dead)
    if not combos:
        return 0.5
    need = 5 - len(board)
    if need == 0:
        mine = evaluate_ints(list(hero) + list(board))
        won = total = 0.0
        for a, b, w in combos:
            theirs = evaluate_ints([a, b] + list(board))
            total += w
            if mine > theirs:
                won += w
            elif mine == theirs:
                won += 0.5 * w
        return won / total if total else 0.5

    deck = [c for c in range(52) if c not in dead]
    cum = []
    run = 0.0
    for c in combos:
        run += c[2]
        cum.append(run)
    picks = rng.choices(combos, cum_weights=cum, k=samples)
    hero = list(hero)
    board = list(board)
    won = 0.0
    done = 0
    for a, b, _ in picks:
        draw = rng.sample(deck, need + 2)
        rest = [c for c in draw if c != a and c != b][:need]
        full = board + rest
        mine = evaluate_ints(hero + full)
        theirs = evaluate_ints([a, b] + full)
        if mine > theirs:
            won += 1.0
        elif mine == theirs:
            won += 0.5
        done += 1
        if deadline is not None and done % 64 == 0 \
                and time.perf_counter() > deadline:
            break
    return won / done if done else 0.5
