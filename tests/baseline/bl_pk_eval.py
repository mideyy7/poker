"""
Fast hand evaluator. Cards are ints 0..51: rank_index * 4 + suit, where
rank_index is 0 for a 2 and 12 for an ace. evaluate_ints takes 5 to 7 cards
and returns one int: higher wins, equal splits (same order as
card_utils.evaluate).
"""

from card_utils import RANKS, SUITS

CAT_SHIFT = 20


def _straight_table():
    """For each 13-bit rank mask, 1 + the rank index of the top card of the
    best straight in it (0 if none). The wheel counts as 5-high."""
    table = [0] * 8192
    for mask in range(8192):
        best = 0
        for top in range(12, 3, -1):
            need = 0x1F << (top - 4)
            if mask & need == need:
                best = top + 1
                break
        if not best and mask & 0x100F == 0x100F:  # A,2,3,4,5
            best = 3 + 1  # top card is the 5 (index 3)
        table[mask] = best
    return table


STRAIGHT = _straight_table()


def _top_bits(mask, n):
    """The n highest set rank indices in mask, high first."""
    out = []
    r = 12
    while r >= 0 and len(out) < n:
        if mask >> r & 1:
            out.append(r)
        r -= 1
    return out


def evaluate_ints(cards):
    counts = [0] * 13
    suit_masks = [0, 0, 0, 0]
    rank_mask = 0
    for c in cards:
        r = c >> 2
        counts[r] += 1
        bit = 1 << r
        suit_masks[c & 3] |= bit
        rank_mask |= bit

    # Flush / straight flush.
    for sm in suit_masks:
        if sm.bit_count() >= 5:
            sf = STRAIGHT[sm]
            if sf:
                return (8 << CAT_SHIFT) | sf
            k = _top_bits(sm, 5)
            return ((5 << CAT_SHIFT) | (k[0] << 16) | (k[1] << 12)
                    | (k[2] << 8) | (k[3] << 4) | k[4])

    quad = -1
    trips = []
    pairs = []
    singles = []
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 0:
            continue
        if n == 4:
            quad = r
        elif n == 3:
            trips.append(r)
        elif n == 2:
            pairs.append(r)
        else:
            singles.append(r)

    if quad >= 0:
        kick = -1
        for r in range(12, -1, -1):
            if counts[r] and r != quad:
                kick = r
                break
        return (7 << CAT_SHIFT) | (quad << 4) | kick
    if trips and (len(trips) > 1 or pairs):
        t = trips[0]
        p = trips[1] if len(trips) > 1 else -1
        if pairs and pairs[0] > p:
            p = pairs[0]
        return (6 << CAT_SHIFT) | (t << 4) | p
    st = STRAIGHT[rank_mask]
    if st:
        return (4 << CAT_SHIFT) | st
    if trips:
        k = singles[:2]
        return (3 << CAT_SHIFT) | (trips[0] << 8) | (k[0] << 4) | k[1]
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kick = -1
        for r in range(12, -1, -1):
            if counts[r] and r != p1 and r != p2:
                kick = r
                break
        return (2 << CAT_SHIFT) | (p1 << 8) | (p2 << 4) | kick
    if pairs:
        k = singles[:3]
        return ((1 << CAT_SHIFT) | (pairs[0] << 12) | (k[0] << 8)
                | (k[1] << 4) | k[2])
    k = singles[:5]
    return ((k[0] << 16) | (k[1] << 12) | (k[2] << 8) | (k[3] << 4) | k[4])


def card_to_int(card):
    return RANKS.index(card.rank) * 4 + SUITS.index(card.suit)


def cards_to_ints(cards):
    return [card_to_int(c) for c in cards]


def int_to_str(c):
    return RANKS[c >> 2] + SUITS[c & 3]


def parse_ints(text):
    """'AS KD' -> ints, for tests."""
    return [RANKS.index(t[0]) * 4 + SUITS.index(t[1]) for t in text.split()]


def category(score):
    return score >> CAT_SHIFT
