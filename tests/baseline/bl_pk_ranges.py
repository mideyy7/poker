"""
Ranges.

Preflop a range is a list of 169 weights, one per starting-hand class
(pk_preflop.CLASSES). Postflop it is a list of (card_a, card_b, weight)
combos. Narrowing keeps the strongest share of a range, with a small floor
weight elsewhere so a range never becomes empty and bluffs stay possible.
"""

from tests.baseline.bl_pk_eval import STRAIGHT, evaluate_ints
from tests.baseline.bl_pk_preflop import CLASSES, EQ

RANKS = "23456789TJQKA"
N = len(CLASSES)
INDEX = {name: i for i, name in enumerate(CLASSES)}


def _combos(name):
    hi, lo = RANKS.index(name[0]), RANKS.index(name[1])
    out = []
    for s1 in range(4):
        for s2 in range(4):
            a, b = hi * 4 + s1, lo * 4 + s2
            if hi == lo:
                if s1 < s2:
                    out.append((a, b))
            elif name[2] == "s" and s1 == s2:
                out.append((a, b))
            elif name[2] == "o" and s1 != s2:
                out.append((a, b))
    return out


CLASS_COMBOS = [_combos(name) for name in CLASSES]
CLASS_SIZE = [len(c) for c in CLASS_COMBOS]

# Equity vs a random hand, and the classes strongest first by it.
VS_RANDOM = [
    sum(EQ[i][j] * CLASS_SIZE[j] for j in range(N)) / 1326.0
    for i in range(N)
]
ORDER = sorted(range(N), key=lambda i: -VS_RANDOM[i])

# PERCENTILE[i]: share of all combos at least as strong as class i
# (0 = best hand, 1 = worst).
PERCENTILE = [0.0] * N
_acc = 0
for _i in ORDER:
    _acc += CLASS_SIZE[_i]
    PERCENTILE[_i] = _acc / 1326.0


def class_of(a, b):
    ra, rb = a >> 2, b >> 2
    if ra < rb:
        ra, rb, a, b = rb, ra, b, a
    if ra == rb:
        return INDEX[RANKS[ra] * 2]
    suited = "s" if (a & 3) == (b & 3) else "o"
    return INDEX[RANKS[ra] + RANKS[rb] + suited]


def full_range():
    return [1.0] * N


def unblocked_sizes(dead):
    """Combos per class that avoid the dead cards."""
    if not dead:
        return CLASS_SIZE
    return [sum(1 for a, b in CLASS_COMBOS[j] if a not in dead and b not in dead)
            for j in range(N)]


def range_share(weights):
    """Share of all 1326 combos this range holds (weighted)."""
    return sum(weights[j] * CLASS_SIZE[j] for j in range(N)) / 1326.0


def narrow_classes(weights, keep, floor=0.03, passive_top=0.0,
                   passive_weight=1.0):
    """
    Keep the strongest `keep` share of the current range (by mass) at full
    weight, the rest at `floor`. If passive_top > 0, the strongest
    passive_top share (hands that would have raised instead) is scaled by
    passive_weight.
    """
    keep = max(0.0, min(1.0, keep))
    total = sum(weights[j] * CLASS_SIZE[j] for j in range(N))
    if total <= 0:
        return full_range()
    out = list(weights)
    acc = 0.0
    for j in ORDER:
        m = weights[j] * CLASS_SIZE[j]
        if m <= 0:
            continue
        start = acc / total
        acc += m
        end = acc / total
        mid = (start + end) / 2
        if mid <= keep:
            f = 1.0
        else:
            f = floor
        if passive_top > 0 and mid <= passive_top:
            f *= passive_weight
        out[j] = weights[j] * f
    return out


def preflop_equity(hero, weights):
    """Equity of the hero's two cards vs a class-weighted range, with
    card removal in combo counts (class-vs-class equities)."""
    h = class_of(hero[0], hero[1])
    sizes = unblocked_sizes(set(hero))
    row = EQ[h]
    num = den = 0.0
    for j in range(N):
        m = weights[j] * sizes[j]
        if m > 0:
            num += m * row[j]
            den += m
    return num / den if den else 0.5


def to_combos(weights, dead):
    """Expand a class-weighted range into live combos."""
    out = []
    for j in range(N):
        w = weights[j]
        if w <= 0:
            continue
        for a, b in CLASS_COMBOS[j]:
            if a not in dead and b not in dead:
                out.append((a, b, w))
    return out


# Strength floors for draws, as evaluate_ints scores.
_PAIR_MID = (1 << 20) | (6 << 12)  # about a pair of eights
_TWO_PAIR = 2 << 20


def combo_strength(a, b, board):
    """Made-hand score on this board, bumped for strong draws before the
    river so semi-bluffs sit mid-range rather than at the bottom."""
    cards = [a, b] + list(board)
    score = evaluate_ints(cards)
    if len(board) >= 5 or score >= _TWO_PAIR:
        return score
    # Flush draw using a hole card.
    fd = False
    for s in {a & 3, b & 3}:
        if sum(1 for c in cards if c & 3 == s) == 4:
            fd = True
    # Straight draw: at least two ranks complete a straight that uses a
    # hole card's rank.
    mask = 0
    for c in cards:
        mask |= 1 << (c >> 2)
    outs = 0
    if not STRAIGHT[mask]:
        for r in range(13):
            bit = 1 << r
            if not mask & bit and STRAIGHT[mask | bit]:
                outs += 1
    oesd = outs >= 2
    if fd and oesd:
        return max(score, _TWO_PAIR)
    if fd or oesd:
        return max(score, _PAIR_MID)
    return score


def narrow_combos(combos, board, keep, floor=0.1, passive_top=0.0,
                  passive_weight=1.0):
    """Postflop version of narrow_classes, ranking combos by
    combo_strength on this board."""
    if not combos:
        return combos
    keep = max(0.0, min(1.0, keep))
    scored = sorted(combos, key=lambda c: -combo_strength(c[0], c[1], board))
    total = sum(c[2] for c in scored)
    if total <= 0:
        return combos
    out = []
    acc = 0.0
    for a, b, w in scored:
        start = acc / total
        acc += w
        mid = (start + acc / total) / 2
        f = 1.0 if mid <= keep else floor
        if passive_top > 0 and mid <= passive_top:
            f *= passive_weight
        out.append((a, b, w * f))
    return out
