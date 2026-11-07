"""
Hand replay and opponent statistics.

replay() walks a hand's actions and labels every player move with the
situation it was made in. Stats keeps, per team, smoothed frequencies of
each move per situation: a prior worth PRIOR_WEIGHT hands, so a few hands
barely move it and a few hundred dominate it.
"""

from pk_eval import card_to_int

STREETS = ("preflop", "flop", "turn", "river")

# Situation -> prior frequency of each move (a population guess).
PRIORS = {
    # Small blind's first move.
    "sb_open": {"fold": 0.15, "limp": 0.25, "raise": 0.50, "jam": 0.10},
    # Big blind after a limp.
    "bb_vs_limp": {"check": 0.70, "raise": 0.25, "jam": 0.05},
    # Big blind facing a (non all-in) open raise.
    "bb_vs_raise": {"fold": 0.35, "call": 0.45, "raise": 0.12, "jam": 0.08},
    # Facing a re-raise (not all-in) preflop.
    "vs_3bet": {"fold": 0.40, "call": 0.35, "raise": 0.10, "jam": 0.15},
    # Facing an all-in, before or after putting more than a blind in.
    "vs_jam_cold": {"fold": 0.55, "call": 0.45},
    "vs_jam_invested": {"fold": 0.35, "call": 0.65},
    # Postflop, nothing to call / facing a bet.
    "post_first": {"check": 0.55, "bet": 0.45},
    "post_facing": {"fold": 0.40, "call": 0.45, "raise": 0.15},
    # Item 3: first bet postflop split by size bucket.
    "post_first_sized": {"check": 0.55, "bet_s": 0.10, "bet_m": 0.25,
                         "bet_l": 0.10},
    # Item 2: big bets that reached showdown, weak hand (bluff) or not.
    "sd_bigbet": {"weak": 0.25, "strong": 0.75},
}

# Item 4: big blind vs limp prior from 25bb solver charts.
SOLVER_BB_VS_LIMP = {"check": 0.57, "raise": 0.37, "jam": 0.06}
PRIOR_WEIGHT = 10.0
DECAY = 0.985  # item 1: memory of about 67 observations per situation


class Event:
    """One player move with its context. Plain data (pickles)."""

    def __init__(self, player, street, situation, move, amount, board_len,
                 pot, to_call):
        self.player = player
        self.street = street
        self.situation = situation
        self.move = move
        self.amount = amount
        self.board_len = board_len
        self.pot = pot
        self.to_call = to_call


def replay(actions, button, small_blind, big_blind, start_stack,
           num_players=2):
    """
    Labels every player move in actions (a full hand from its first move).
    Returns (events, board_ints, shown) where shown maps player -> hole
    card ints revealed at showdown.
    """
    other = 1 - button if num_players == 2 else (button + 1) % num_players
    bets = [0] * num_players
    behind = [start_stack] * num_players
    bets[button] = small_blind
    bets[other] = big_blind
    behind[button] -= small_blind
    behind[other] -= big_blind
    pot = small_blind + big_blind
    current = big_blind
    raises = 0  # raises this street
    raised_before = [False] * num_players  # put in a voluntary raise
    board = []
    shown = {}
    events = []
    for act in actions:
        kind = act.kind
        p = act.player_id
        if kind == "reveal_card":
            card = card_to_int(act.card)
            if p < 0:
                board.append(card)
                if len(board) in (3, 4, 5):
                    bets = [0] * num_players
                    current = 0
                    raises = 0
            else:
                shown.setdefault(p, []).append(card)
            continue
        if kind == "win" or p < 0 or p >= num_players:
            continue
        street = STREETS[{0: 0, 3: 1, 4: 2, 5: 3}.get(len(board), 3)]
        to_call = max(0, current - bets[p])
        aggressor_all_in = to_call > 0 and any(
            behind[q] == 0 and bets[q] == current
            for q in range(num_players) if q != p)
        all_in_amount = bets[p] + behind[p]

        # Situation.
        if aggressor_all_in:
            invested = raised_before[p] or (street == "preflop"
                                            and bets[p] > big_blind)
            situation = "vs_jam_invested" if invested else "vs_jam_cold"
        elif street == "preflop":
            if p == button and raises == 0 and bets[p] == small_blind \
                    and current == big_blind:
                situation = "sb_open"
            elif p != button and raises == 0:
                situation = "bb_vs_limp"
            elif raises == 1 and not raised_before[p]:
                situation = "bb_vs_raise" if p != button else "vs_3bet"
            else:
                situation = "vs_3bet"
        else:
            situation = "post_facing" if to_call > 0 else "post_first"

        # Move.
        if kind == "fold":
            move = "fold" if to_call > 0 else "check"
        elif kind == "call":
            if to_call == 0:
                move = "check"
            elif situation == "sb_open":
                move = "limp"
            else:
                move = "call"
        else:  # raise
            jam = act.amount >= all_in_amount
            if situation == "post_first":
                move = "bet"
            elif situation.startswith("vs_jam"):
                move = "call"
            elif jam and situation in ("sb_open", "bb_vs_limp",
                                       "bb_vs_raise", "vs_3bet"):
                move = "jam"
            else:
                move = "raise"

        events.append(Event(p, street, situation, move,
                            act.amount, len(board), pot, to_call))

        # Apply.
        if kind == "call":
            put = min(to_call, behind[p])
            bets[p] += put
            behind[p] -= put
            pot += put
        elif kind == "raise":
            put = min(act.amount - bets[p], behind[p])
            bets[p] += put
            behind[p] -= put
            pot += put
            if bets[p] > current:
                current = bets[p]
                raises += 1
                raised_before[p] = True
    return events, board, shown


class Stats:
    """Per-team move counts per situation. Plain dicts only.

    decay: old evidence fades by this factor per new observation in the
    same situation (1.0 = never forget). priors: situation -> move -> prior
    frequency (defaults to PRIORS). weight: per-team multiplier on the
    prior weight (more = trust the prior more, exploit less).
    """

    def __init__(self, decay=1.0, priors=None):
        self.counts = {}  # team -> situation -> move -> count
        self.hands = {}  # team -> hands seen
        self.decay = decay
        self.priors = {k: dict(v) for k, v in (priors or PRIORS).items()}
        self.weight = {}  # team -> prior weight multiplier

    def add(self, team, situation, move):
        sit = self.counts.setdefault(team, {}).setdefault(situation, {})
        if self.decay < 1.0:
            for m in sit:
                sit[m] *= self.decay
        sit[move] = sit.get(move, 0) + 1

    def seen_hand(self, team):
        self.hands[team] = self.hands.get(team, 0) + 1

    def n(self, team, situation):
        return sum(self.counts.get(team, {}).get(situation, {}).values())

    def freq(self, team, situation, move):
        prior = self.priors[situation]
        sit = self.counts.get(team, {}).get(situation, {})
        total = sum(sit.values())
        w = PRIOR_WEIGHT * self.weight.get(team, 1.0)
        return (sit.get(move, 0) + w * prior.get(move, 0.0)) / (total + w)

    def freqs(self, team, situation):
        return {m: self.freq(team, situation, m)
                for m in self.priors[situation]}

    def learn(self, team, player, events):
        for e in events:
            if e.player == player and e.situation in self.priors                     and e.move in self.priors[e.situation]:
                self.add(team, e.situation, e.move)
            # Bet sizes when first to bet postflop (item 3).
            if e.player == player and e.situation == "post_first":
                self.add(team, "post_first_sized",
                         "check" if e.move != "bet"
                         else "bet_" + size_bucket(e.amount, e.pot))


def size_bucket(amount, pot):
    """s: under 0.4 pot, l: 0.9 pot or more, m: between."""
    ratio = amount / max(pot, 1)
    if ratio < 0.4:
        return "s"
    if ratio >= 0.9:
        return "l"
    return "m"
