import random
import time

from card_utils import Card
from game_state import GameStateView
from player_action import PlayerAction

from pk_equity import equity
from pk_eval import cards_to_ints, category, evaluate_ints
from pk_opp import (
    DECAY,
    PRIORS,
    SOLVER_BB_VS_LIMP,
    Stats,
    replay,
    size_bucket,
)
from pk_ranges import (
    VS_RANDOM,
    class_of,
    full_range,
    narrow_classes,
    narrow_combos,
    preflop_equity,
    to_combos,
)

# Share of the pot we expect to realise when calling and playing on,
# out of / in position (crude, but better than assuming showdown).
REALISE_OOP = 0.85
REALISE_IP = 0.95
TIME_BUDGET = 1.0  # seconds of sampling per decision; the limit is 10 s

# Improvements from strong.md, each switchable so it can be gated alone.
ALL_FEATURES = ("decay", "showdown", "sizes", "open2", "raise_term",
                "loss_limit", "adaptive")
# open2 (item 4) is off: the gate showed it gives up a limp-and-value-bet
# exploit vs bots that fold preflop but call postflop (nit, -70 bb/100).
DEFAULT_FEATURES = tuple(f for f in ALL_FEATURES if f != "open2")


class Model:
    """
    Heads-up NLHE bot for 25bb stacks.

    Preflop: picks the move with the best expected chips given the
    opponent's observed frequencies (fold / call / raise / jam per
    situation). Postflop: equity vs the opponent's range, narrowed by
    their actions, with fold equity from their observed fold rate.
    """

    def __init__(self, samples: int = 1200, features=DEFAULT_FEATURES):
        self.hand = []
        self.player_id = None
        self.hole = []
        self.history = []
        self.samples = samples
        self.features = {f: f in features for f in ALL_FEATURES}
        priors = {k: dict(v) for k, v in PRIORS.items()}
        if self.features["open2"]:
            priors["bb_vs_limp"] = dict(SOLVER_BB_VS_LIMP)
        self.stats = Stats(decay=DECAY if self.features["decay"] else 1.0,
                           priors=priors)
        self.start_stack = 500
        self.errors = 0  # decisions that fell back to check/fold
        self.results = {}  # team -> [hands, sum, sum of squares] (chips)
        self.cur_pot = 0

    # ------------------------------------------------------------------
    # Engine hooks

    def seed(self, hand: list[Card], player_id: int):
        self.hand = hand
        self.player_id = player_id
        self.hole = cards_to_ints(hand)
        self.history = []

    def run(self, actions: list[PlayerAction],
            game_state: GameStateView) -> PlayerAction:
        self.history.extend(actions)
        try:
            action = self._decide(game_state)
            return self._legal(action, game_state)
        except Exception:
            self.errors += 1
            return self._check_or_fold(game_state)

    def end_game(self, actions: list[PlayerAction],
                 game_state: GameStateView):
        self.history.extend(actions)
        try:
            if game_state.num_players != 2:
                return
            opp = 1 - self.player_id
            team = game_state.teams[opp]
            events, board, shown = replay(
                self.history, game_state.button, game_state.small_blind,
                game_state.big_blind, self.start_stack)
            self.stats.learn(team, opp, events)
            self.stats.seen_hand(team)
            if self.features["showdown"] and opp in shown                     and len(board) == 5:
                self._learn_showdown(team, opp, events, board, shown[opp])
            if self.features["loss_limit"]:
                self._track_result(team, game_state)
        except Exception:
            pass

    def _learn_showdown(self, team, opp, events, board, hole):
        """Item 2: was their big postflop bet or raise a bluff? Weak means
        their hand did not beat the board's own category."""
        big = [e for e in events if e.player == opp and e.street != "preflop"
               and e.move in ("bet", "raise") and e.amount >= 0.6 * e.pot]
        if not big:
            return
        mine = category(evaluate_ints(list(hole) + board))
        weak = mine <= category(evaluate_ints(board))
        self.stats.add(team, "sd_bigbet", "weak" if weak else "strong")

    def _track_result(self, team, gs):
        """Item 6: our running result vs this team; if we are losing
        significantly, trust the population prior more (exploit less)."""
        net = gs.stacks[self.player_id] - self.start_stack
        r = self.results.setdefault(team, [0, 0.0, 0.0])
        r[0] += 1
        r[1] += net
        r[2] += net * net
        n = r[0]
        if n < 150:
            return
        mean = r[1] / n
        var = max(r[2] / n - mean * mean, 1.0)
        se = (var / n) ** 0.5
        self.stats.weight[team] = 4.0 if mean + 2 * se < 0 else 1.0

    # ------------------------------------------------------------------
    # Helpers

    def _check_or_fold(self, gs):
        if gs.call_amount == 0:
            return PlayerAction.call(self.player_id)
        return PlayerAction.fold(self.player_id)

    def _call(self):
        return PlayerAction.call(self.player_id)

    def _fold_or_check(self, gs):
        return self._check_or_fold(gs)

    def _raise(self, gs, total):
        if not gs.can_raise:
            return PlayerAction.call(self.player_id)
        low, high = gs.raise_range
        total = int(round(total))
        # Raising most of the stack: just go all-in.
        if total >= 0.8 * high:
            total = high
        return PlayerAction.raise_to(self.player_id, min(high, max(low, total)))

    def _legal(self, action, gs):
        """Last line of defence: never return an illegal move."""
        if action.kind == "fold" and gs.call_amount == 0:
            return PlayerAction.call(self.player_id)
        if action.kind == "raise":
            if not gs.can_raise:
                return PlayerAction.call(self.player_id)
            low, high = gs.raise_range
            if not low <= action.amount <= high:
                return PlayerAction.raise_to(
                    self.player_id, min(high, max(low, action.amount)))
        return action

    def _rng(self, gs):
        seed = sum(c * 53 ** i for i, c in enumerate(self.hole))
        seed = seed * 131 + len(gs.board) * 7 + len(self.history)
        return random.Random(seed)

    # ------------------------------------------------------------------
    # Decision

    def _decide(self, gs):
        self.deadline = time.perf_counter() + TIME_BUDGET
        self.cur_pot = gs.pot
        n = gs.num_players
        self.start_stack = max(1, (sum(gs.stacks) + gs.pot) // n)
        if n != 2:
            return self._multiway(gs)
        me = self.player_id
        opp = 1 - me
        team = gs.teams[opp]
        events, board, _ = replay(self.history, gs.button, gs.small_blind,
                                  gs.big_blind, self.start_stack)
        weights = self._villain_preflop_range(events, opp, team)
        if gs.street == "preflop":
            return self._preflop(gs, me, opp, team, weights)
        board = cards_to_ints(gs.board)
        combos = self._villain_postflop_range(events, opp, team, weights,
                                              board)
        return self._postflop(gs, me, opp, team, board, combos)

    # -- ranges ---------------------------------------------------------

    def _villain_preflop_range(self, events, opp, team):
        w = full_range()
        for e in events:
            if e.player != opp or e.street != "preflop":
                continue
            f = self.stats.freqs(team, e.situation)
            agg = f.get("raise", 0.0) + f.get("jam", 0.0)
            if e.move == "jam":
                keep = f.get("jam", 0.0) + 0.5 * f.get("raise", 0.0)
                w = narrow_classes(w, max(keep, 0.04), floor=0.02)
            elif e.move == "raise":
                w = narrow_classes(w, max(agg, 0.04), floor=0.03)
            elif e.move in ("call", "limp"):
                cont = 1.0 - f.get("fold", 0.0)
                if e.situation.startswith("vs_jam"):
                    w = narrow_classes(w, max(cont, 0.04), floor=0.02)
                else:
                    w = narrow_classes(w, max(cont, 0.05), floor=0.03,
                                       passive_top=agg, passive_weight=0.35)
            elif e.move == "check":
                w = narrow_classes(w, 1.0, passive_top=agg,
                                   passive_weight=0.35)
        return w

    def _villain_postflop_range(self, events, opp, team, weights, board):
        combos = to_combos(weights, set(self.hole) | set(board))
        for e in events:
            if e.player != opp or e.street == "preflop":
                continue
            b = board[:e.board_len]
            f = self.stats.freqs(team, e.situation)
            if e.situation == "post_first":
                if e.move == "bet":
                    keep = f["bet"]
                    top = 0.0
                    if self.features["sizes"]:
                        # Item 3: larger buckets hold the stronger hands.
                        fs = self.stats.freqs(team, "post_first_sized")
                        bucket = size_bucket(e.amount, e.pot)
                        bigger = {"s": ("bet_m", "bet_l"), "m": ("bet_l",),
                                  "l": ()}[bucket]
                        top = sum(fs[k] for k in bigger)
                        keep = top + fs["bet_" + bucket]
                    elif e.amount > e.pot:  # overbet: stronger
                        keep *= 0.8
                    floor = self._bluff_floor(team, keep, e) or 0.2
                    combos = narrow_combos(combos, b, keep, floor=floor,
                                           passive_top=top,
                                           passive_weight=0.3)
                else:
                    combos = narrow_combos(combos, b, 1.0,
                                           passive_top=f["bet"] * 0.6,
                                           passive_weight=0.5)
            elif e.situation == "post_facing":
                if e.move == "call":
                    combos = narrow_combos(combos, b, 1.0 - f["fold"],
                                           floor=0.1, passive_top=f["raise"],
                                           passive_weight=0.5)
                elif e.move == "raise":
                    floor = self._bluff_floor(team, f["raise"], e) or 0.1
                    combos = narrow_combos(combos, b, f["raise"], floor=floor)
            elif e.situation.startswith("vs_jam") and e.move == "call":
                combos = narrow_combos(combos, b, f["call"], floor=0.05)
        return combos

    def _bluff_floor(self, team, keep, e):
        """Item 2: weight on hands outside the top `keep` so that their
        share of a big bet's range equals the bluff share seen at
        showdown. None when the feature is off or the bet is small."""
        if not self.features["showdown"] or e.amount < 0.6 * e.pot:
            return None
        bluff = self.stats.freq(team, "sd_bigbet", "weak")
        keep = min(max(keep, 0.02), 0.98)
        floor = bluff * keep / ((1 - bluff) * (1 - keep))
        return min(1.0, max(0.02, floor))

    # -- preflop --------------------------------------------------------

    def _preflop(self, gs, me, opp, team, w):
        hole = self.hole
        bb = gs.big_blind
        my_bet = gs.bets[me]
        my_total = gs.stacks[me] + my_bet
        opp_total = gs.stacks[opp] + gs.bets[opp]
        eff = min(my_total, opp_total)
        pot = gs.pot
        call = gs.call_amount
        stats = self.stats
        ip = me == gs.button  # SB is in position postflop

        def jam_ev(fold_situation, rng_w):
            pf = stats.freq(team, fold_situation, "fold")
            call_range = narrow_classes(rng_w, max(1.0 - pf, 0.03),
                                        floor=0.02)
            eq_c = preflop_equity(hole, call_range)
            return pf * pot + (1 - pf) * (eq_c * 2 * eff - (eff - my_bet))

        def call_jam_ev(range_w, invested):
            """EV (from the moment we face it) of the better of folding or
            calling an all-in; `invested` is what we already have in."""
            eq = preflop_equity(hole, range_w)
            return max(-0.0, eq * 2 * eff - (eff - invested))

        # Facing an all-in (or a call that is all we can do).
        opp_all_in = gs.stacks[opp] == 0
        if call > 0 and (opp_all_in or not gs.can_raise
                         or call >= gs.stacks[me]):
            eq = preflop_equity(hole, w)
            excess = max(0, gs.bets[opp] - (my_bet + min(call, gs.stacks[me])))
            final_pot = pot + min(call, gs.stacks[me]) - excess
            if eq * final_pot - min(call, gs.stacks[me]) > 0:
                return self._call()
            return self._fold_or_check(gs)

        # Small blind, first move.
        if me == gs.button and call == gs.big_blind - gs.small_blind \
                and gs.current_bet == bb and my_bet == gs.small_blind:
            return self._sb_open(gs, team, w, eff, jam_ev, call_jam_ev)

        # Big blind after a limp.
        if call == 0:
            return self._bb_vs_limp(gs, team, w, eff, jam_ev, call_jam_ev)

        # Facing a raise (not all-in): fold, call or jam.
        eq = preflop_equity(hole, w)
        realise = REALISE_IP if ip else REALISE_OOP
        if gs.stacks[me] - call < 1.5 * (pot + call):
            realise = 1.0  # committed: close to showdown
        ev_call = eq * (pot + call) * realise - call
        ev_jam = jam_ev("vs_jam_invested", w) if gs.can_raise else -1e9
        if ev_jam >= ev_call and ev_jam > 0:
            return self._raise(gs, gs.max_raise_to)
        if ev_call > 0:
            return self._call()
        return self._fold_or_check(gs)

    def _sb_open(self, gs, team, w, eff, jam_ev, call_jam_ev):
        hole = self.hole
        bb, sb = gs.big_blind, gs.small_blind
        pot = gs.pot
        stats = self.stats

        # Raise to 2.5bb, or a min-raise vs a bot that folds to raises.
        f_bb = stats.freqs(team, "bb_vs_raise")
        call_range = narrow_classes(full_range(), 1 - f_bb["fold"],
                                    passive_top=f_bb["raise"] + f_bb["jam"],
                                    passive_weight=0.35)
        eq_called = preflop_equity(hole, call_range)
        reraise_range = narrow_classes(full_range(),
                                       max(f_bb["raise"] + f_bb["jam"], 0.04),
                                       floor=0.02)

        def open_ev(size):
            cost = size - sb
            return (
                f_bb["fold"] * pot
                + f_bb["call"] * (eq_called * 2 * size * REALISE_IP - cost)
                + (f_bb["raise"] + f_bb["jam"])
                * (call_jam_ev(reraise_range, size) - cost)
            )

        if self.features["open2"]:
            # Item 4: let the EV choose between a 2bb and a 2.5bb open.
            size = max((2 * bb, int(2.5 * bb)), key=open_ev)
        else:
            size = 2 * bb if f_bb["fold"] > 0.6 else int(2.5 * bb)
        ev_raise = open_ev(size)

        # Limp.
        f_l = stats.freqs(team, "bb_vs_limp")
        attack = f_l["raise"] + f_l["jam"]
        check_range = narrow_classes(full_range(), 1.0, passive_top=attack,
                                     passive_weight=0.35)
        eq_check = preflop_equity(hole, check_range)
        attack_range = narrow_classes(full_range(), max(attack, 0.04),
                                      floor=0.02)
        limp_cost = bb - sb
        ev_limp = (
            f_l["check"] * (eq_check * 2 * bb * REALISE_IP - limp_cost)
            + attack * (call_jam_ev(attack_range, bb) - limp_cost)
        )

        ev_jam = jam_ev("vs_jam_cold", full_range())

        best = max(("fold", 0.0), ("limp", ev_limp), ("raise", ev_raise),
                   ("jam", ev_jam), key=lambda x: x[1])[0]
        if best == "raise":
            return self._raise(gs, size)
        if best == "jam":
            return self._raise(gs, gs.max_raise_to)
        if best == "limp":
            return self._call()
        return self._fold_or_check(gs)

    def _bb_vs_limp(self, gs, team, w, eff, jam_ev, call_jam_ev):
        hole = self.hole
        bb = gs.big_blind
        pot = gs.pot
        eq = preflop_equity(hole, w)
        ev_check = eq * pot * REALISE_OOP
        if not gs.can_raise:
            return self._call()
        size = 4 * bb
        cost = size - bb
        f = self.stats.freqs(team, "vs_3bet")
        call_range = narrow_classes(w, max(1 - f["fold"], 0.05),
                                    passive_top=f["raise"] + f["jam"],
                                    passive_weight=0.35)
        eq_c = preflop_equity(hole, call_range)
        rr_range = narrow_classes(w, max(f["raise"] + f["jam"], 0.04),
                                  floor=0.02)
        ev_raise = (
            f["fold"] * pot
            + f["call"] * (eq_c * 2 * size * REALISE_OOP - cost)
            + (f["raise"] + f["jam"]) * (call_jam_ev(rr_range, size) - cost)
        )
        ev_jam = jam_ev("vs_jam_cold", w)
        best = max(("check", ev_check), ("raise", ev_raise),
                   ("jam", ev_jam), key=lambda x: x[1])[0]
        if best == "raise":
            return self._raise(gs, size)
        if best == "jam":
            return self._raise(gs, gs.max_raise_to)
        return self._call()

    # -- postflop -------------------------------------------------------

    def _eq(self, board, combos, rng, samples=None):
        n = samples or self.samples
        if self.features["adaptive"] and self.cur_pot >= 200:
            n = int(n * 1.5)  # item 7: big pots deserve accurate equity
        return equity(self.hole, board, combos, n, rng, self.deadline)

    def _postflop(self, gs, me, opp, team, board, combos):
        rng = self._rng(gs)
        pot = gs.pot
        self.cur_pot = pot
        call = gs.call_amount
        behind = gs.stacks[me]
        eff = gs.effective_stack(me)
        river = gs.street == "river"
        ip = me == gs.button
        realise = 1.0 if river else (REALISE_IP if ip else REALISE_OOP)
        stats = self.stats
        f = stats.freqs(team, "post_facing")
        fold0 = f["fold"]

        eq = self._eq(board, combos, rng)

        if call > 0:
            paid = min(call, behind)
            if paid >= behind:
                realise = 1.0
            ev_call = eq * (pot + paid) * realise - paid
            best, best_ev = "fold", 0.0
            if ev_call > best_ev:
                best, best_ev = "call", ev_call
            if gs.can_raise and (eq > 0.55 or fold0 > 0.6):
                to = gs.current_bet * 3 + (pot - gs.current_bet - gs.bets[me])
                if behind + gs.bets[me] - to < 0.6 * (pot + to):
                    to = gs.max_raise_to
                to = min(gs.max_raise_to, max(gs.min_raise_to, int(to)))
                added = to - gs.bets[me]
                opp_add = min(to, gs.stacks[opp] + gs.bets[opp]) \
                    - gs.bets[opp]
                fp = fold0
                cont = narrow_combos(combos, board, max(1 - fp, 0.05),
                                     floor=0.05)
                eq_c = self._eq(board, cont, rng)
                ev_raise = fp * pot + (1 - fp) * (
                    eq_c * (pot + added + opp_add) - added)
                if ev_raise > best_ev:
                    best, best_ev = "raise", ev_raise
                    raise_to = to
            if best == "raise":
                return self._raise(gs, raise_to)
            if best == "call":
                return self._call()
            return self._fold_or_check(gs)

        # Nothing to call: check or bet.
        ev_check = eq * pot * realise
        if not gs.can_raise or eff <= 0:
            return self._call()
        sizes = {int(0.5 * pot), int(0.8 * pot)}
        if eff <= 2.0 * pot:
            sizes.add(eff)
        best, best_ev = None, ev_check
        for b in sorted(sizes):
            b = min(max(b, gs.min_raise_to), gs.max_raise_to)
            ratio = b / max(pot, 1)
            fp = min(0.97, fold0 * (ratio / 0.66) ** 0.35)
            called_add = min(b, gs.stacks[opp])
            if self.features["raise_term"] and b < eff:
                # Item 5: they may raise; we then call or fold, whichever
                # is better vs their raising range (as if all-in).
                fr = min(f["raise"], 1 - fp)
                fc = 1 - fp - fr
                cont = narrow_combos(combos, board, max(1 - fp, 0.05),
                                     floor=0.05, passive_top=fr,
                                     passive_weight=0.3)
                eq_c = self._eq(board, cont, rng,
                                samples=self.samples // 2 + 100)
                raisers = narrow_combos(combos, board, max(fr, 0.03),
                                        floor=0.05)
                eq_r = self._eq(board, raisers, rng, samples=300)
                vs_raise = max(-b, eq_r * (pot + 2 * eff) - eff)
                ev = (fp * pot + fc * (eq_c * (pot + b + called_add) - b)
                      + fr * vs_raise)
            else:
                cont = narrow_combos(combos, board, max(1 - fp, 0.05),
                                     floor=0.05)
                eq_c = self._eq(board, cont, rng,
                                samples=self.samples // 2 + 100)
                ev = fp * pot + (1 - fp) * (eq_c * (pot + b + called_add) - b)
            if ev > best_ev + 0.02 * pot:
                best, best_ev = b, ev
        if best is not None:
            return self._raise(gs, best)
        return self._call()

    # -- more than two players (not the tournament format) ---------------

    def _multiway(self, gs):
        rng = self._rng(gs)
        board = cards_to_ints(gs.board)
        opponents = max(1, len(gs.players_in_hand) - 1)
        if gs.street == "preflop":
            eq = VS_RANDOM[class_of(self.hole[0], self.hole[1])]
        else:
            combos = to_combos(full_range(), set(self.hole) | set(board))
            eq = self._eq(board, combos, rng, samples=300)
        eq = eq ** opponents
        if eq > 0.6 and gs.can_raise:
            return self._raise(gs, gs.current_bet + gs.pot)
        if eq > gs.pot_odds:
            return self._call()
        return self._check_or_fold(gs)
