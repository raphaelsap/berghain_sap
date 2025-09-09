"""
Solver for the Listen Labs “Berghain Challenge”.

The challenge presents you with an online bouncer game: people arrive one by
one with a set of binary attributes (e.g. local/non‑local, wearing all
black, etc.), and your goal is to admit exactly a target number of
patrons (typically 1 000) while meeting a collection of minimum quota
constraints on those attributes.  For example, one scenario might require
that at least 40 % of the attendees are Berlin locals and at least
80 % are wearing all black.  You win by filling the club with as few
rejections as possible while never violating any of the quotas.  The game
exposes a simple REST API (see the challenge instructions) for creating
a game, fetching the next person in the queue, and recording your accept
or reject decision.

This module implements a generic solver which uses the supplied
attribute frequencies and minimum counts to drive an on‑line admission
strategy.  The solver is *greedy but cautious*: it always admits
someone who helps satisfy an outstanding constraint, but will only admit
someone who doesn’t help if there is strong statistical evidence that
the remaining seats can still be filled while meeting all quotas.

The solver works as follows:

1. When a new game is created (via the ``/new-game`` endpoint), the
   server returns a list of quota constraints and estimates of the
   marginal relative frequency ``p_i`` of each attribute and the
   pairwise correlations between attributes.  For simplicity we treat
   the attributes as independent and use only the marginal probabilities
   when reasoning about how many more positive examples are likely to
   arrive.

2. At every step we keep track of the number of people accepted and
   the number accepted for each attribute.  From this we compute how
   many more positives are required for each attribute to meet the
   minimum quotas.  If the current candidate possesses at least one
   attribute for which we still need more positives, we accept them
   immediately (this avoids burning a slot on a candidate who does not
   help meet any outstanding quota).

3. If the current candidate does *not* help satisfy any remaining
   constraints, we evaluate whether we can safely admit them without
   jeopardising our ability to meet the quotas.  Suppose we have
   ``R`` seats left to fill and we still need ``S_i`` more positives
   for attribute ``i``.  Let ``new_R = R - 1`` and ``S_i'`` be the
   remaining requirement for attribute ``i`` after admitting the
   candidate (which is the same as ``S_i`` if the candidate does not
   possess attribute ``i``).  Under the assumption that future
   candidates are drawn independently with probability ``p_i`` of
   having attribute ``i``, the number of attribute‑``i`` positives
   among the next ``new_R`` accepted attendees is binomial with mean
   ``mu_i = new_R * p_i`` and standard deviation ``sigma_i = sqrt(mu_i
   * (1 - p_i))``.  To account for randomness, we require that

       mu_i - k * sigma_i >= S_i'

   for every attribute ``i`` with an outstanding quota, where ``k``
   controls how many standard deviations of cushion we insist upon.  A
   larger ``k`` makes the solver more conservative (it rejects more
   unhelpful candidates early to build a larger margin), while a
   smaller ``k`` makes it more aggressive.  Empirically, values of
   ``k`` between 2 and 3 work well for the three scenarios provided in
   the challenge.

Usage:

    python berghain_solver.py --base-url <challenge_base_url> \
                              --player-id <your_player_uuid> \
                              --scenario <1|2|3> [--target 1000]

Defaults favor a conservative TUI strategy:
- Policy: chernoff with alpha=0.002
- Online probability updates enabled (prior_weight=300)
- TUI dashboard enabled (refresh every 5 candidates)
- Verbose line logs disabled (use --verbose if desired)

You can switch to the normal k-sigma policy with, e.g., `--policy normal --k 3.3`. Use `--no-tui` to disable the dashboard.

Auto-run optimizer:
- Use `--auto` to loop new games, try parameter configurations, and stop when a goal is met.
- Configure with `--max-games` and `--goal-rejections`.

The script uses the ``requests`` library to talk to the server and prints
diagnostic information as it plays.  It terminates when the game is
completed or fails.  The final rejection count is displayed at the end.

This solver deliberately does not rely on any APIs provided by the
``api_tool`` helper; it uses plain HTTP requests to interact with the
game, as required by the challenge description.
"""

import argparse
import os
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, Set, Tuple, List, Optional

import requests


@dataclass
class GameState:
    """Maintain the state of the game during play."""

    # Constant parameters
    game_id: str
    target: int
    probabilities: Dict[str, float]
    min_counts: Dict[str, int]
    cushion: float
    # Policy and logging controls
    policy: str = "normal"  # one of {"normal", "chernoff"}
    alpha: float = 0.01  # overall failure budget for chernoff
    update_probs: bool = False  # blend observed frequencies with server priors
    prior_weight: float = 200.0  # pseudo-counts for prior when blending
    verbose: bool = False
    log_every: int = 10
    # Global seat reserve guard: require remaining seats to cover
    # sum of outstanding needs plus this slack before admitting unhelpful
    reserve_slack: int = 0
    use_seat_reserve: bool = True
    # New: per-attribute seats-budget guard and severity helpers
    use_seats_budget_guard: bool = True
    k_relax_min: float = 1.7  # min k late in the game (dynamic schedule)
    # Fast policy parameter (simple multiplicative cushion on expected supply)
    gamma: float = 0.85
    # TUI controls
    tui: bool = True
    tui_every: int = 5
    tui_max_rows: int = 12
    history_success: list = field(default_factory=list)
    history_len: int = 60

    # --- Prudence automatique pour quotas rares ---
    # p<threshold => considéré "rare"
    rare_p_thresh: float = 0.10
    # multiplicateur max appliqué à k pour un attribut très rare et très tendu
    rare_boost_max: float = 2.0
    # intensité de l'effet (0.0 = désactivé, 1.0 = par défaut)
    rare_beta: float = 1.0

    # Dynamic state
    accepted_total: int = 0
    rejected_count: int = 0
    accepted_by_attr: Dict[str, int] = field(default_factory=dict)
    person_index: int = 0
    # Observations of the incoming stream (for optional probability updates)
    seen_count: int = 0
    seen_by_attr: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Initialize accepted_by_attr counts
        for attr in self.min_counts:
            self.accepted_by_attr.setdefault(attr, 0)
            self.seen_by_attr.setdefault(attr, 0)

    # ---- Helpers ----
    def current_probability(self, attr: str) -> float:
        """Return a conservative probability for attr (blended, lower-bound).

        We blend prior p0 with observed frequency using pseudo-counts, then
        apply a normal-approx lower bound with k-sigma cushion (dynamic k).
        """
        p0 = float(self.probabilities.get(attr, 0.0))
        # Blended mean
        if self.update_probs and self.seen_count > 0:
            c = float(self.seen_by_attr.get(attr, 0))
            w = float(self.prior_weight)
            n = float(self.seen_count)
            denom = w + n
            p_hat = (w * p0 + c) / denom if denom > 0 else p0
            # Lower bound via normal approx: p_hat - k * sqrt(p_hat(1-p_hat)/denom)
            k_eff = self.effective_k()
            var = p_hat * max(0.0, 1.0 - p_hat)
            se = math.sqrt(var / denom) if denom > 0 else 0.0
            p_lb = p_hat - k_eff * se
            return max(0.0, min(1.0, p_lb))
        else:
            # No observations yet: return prior as-is
            return max(0.0, min(1.0, p0))

    def effective_k(self) -> float:
        """Dynamic cushion: stricter early, relax later.

        Scales between k_max=self.cushion (early) and k_min=self.k_relax_min (late).
        """
        progress = self.accepted_total / max(1, self.target)
        return max(self.k_relax_min, self.cushion - (self.cushion - self.k_relax_min) * progress)

    # ----- RARETÉ : pondération et k par attribut -----
    def rarity_weight(self, p_i: float, s_i: int, R: int) -> float:
        """
        Calcule un poids de rareté/tension dans [1, rare_boost_max].
        - scarcity in [0,1]: plus p_i est petit (< rare_p_thresh), plus c'est rare.
        - pressure in [0,1]: plus la part s_i/R est grande, plus le besoin est tendu.
        """
        if self.rare_beta <= 0.0:
            return 1.0
        if p_i <= 0.0:
            return self.rare_boost_max
        scarcity = max(0.0, min(1.0, (self.rare_p_thresh - p_i) / max(1e-9, self.rare_p_thresh)))
        pressure = 0.0 if R <= 0 else max(0.0, min(1.0, s_i / float(R)))
        raw = 1.0 + self.rare_beta * scarcity * (0.5 + 0.5 * pressure)
        return max(1.0, min(self.rare_boost_max, raw))

    def per_attr_k(self, p_i: float, s_i: int, R: int) -> float:
        """k effectif pour un attribut donné, amplifié si rare/tendu."""
        return self.effective_k() * self.rarity_weight(p_i, s_i, R)

    def seats_required_normal(self, need: int, p: float, k: float, max_n: int) -> int:
        """Smallest n such that n*p - k*sqrt(n*p*(1-p)) >= need. Binary search."""
        if need <= 0:
            return 0
        if p <= 0.0:
            return max_n + 1  # impossible within remaining seats
        lo, hi = 0, max_n
        ans = max_n + 1
        # Binary search over n
        while lo <= hi:
            mid = (lo + hi) // 2
            mu = mid * p
            sigma = math.sqrt(mu * max(0.0, 1.0 - p)) if 0.0 < p < 1.0 else 0.0
            lb = mu - k * sigma
            if lb >= need:
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans

    # ---- Stats helpers ----
    @staticmethod
    def _normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _attr_failure_prob(self, s_i: int, R: int, p_i: float) -> float:
        """Estimate/bound failure probability that X < s_i among next R accepts.

        - Chernoff: bound exp(-mu * (1 - s/mu)^2 / 2) if s < mu else 1.0.
        - Normal: approx Phi((s - mu)/sigma) with continuity correction ignored.
        """
        mu = R * p_i
        if self.policy == "chernoff":
            if mu <= 0.0:
                return 1.0 if s_i > 0 else 0.0
            if s_i >= mu:
                return 1.0
            ratio = max(0.0, min(1.0, s_i / mu))
            delta = 1.0 - ratio
            return math.exp(-mu * (delta ** 2) / 2.0)
        # normal approx
        if s_i <= 0:
            return 0.0
        if p_i <= 0.0:
            return 1.0
        sigma = math.sqrt(mu * (1.0 - p_i)) if 0.0 < p_i < 1.0 else 0.0
        if sigma <= 0.0:
            return 0.0 if mu >= s_i else 1.0
        z = (s_i - mu) / sigma
        return self._normal_cdf(z)

    def _render_bar(self, value: int, total: int, width: int = 40) -> str:
        frac = 0.0 if total <= 0 else max(0.0, min(1.0, value / float(total)))
        filled = int(round(frac * width))
        return "[" + ("█" * filled + " " * (width - filled)) + f"] {value}/{total} ({frac*100:.1f}%)"

    def _render_sparkline(self, values: list, width: int = 40) -> str:
        if not values:
            return ""
        blocks = "▁▂▃▄▅▆▇█"
        # take the last 'width' points
        tail = values[-width:]
        chars = []
        for v in tail:
            v = 0.0 if v is None else max(0.0, min(1.0, float(v)))
            idx = min(len(blocks) - 1, int(round(v * (len(blocks) - 1))))
            chars.append(blocks[idx])
        return "".join(chars)

    def _estimate_success(self) -> Tuple[float, Dict[str, float], Tuple[str, float]]:
        """Compute overall success estimate and per-attribute failure probs.

        Returns (overall_estimate, per_attr_fail, tightest_pair).
        overall_estimate is a lower bound with Chernoff or an approximation with normal.
        """
        needed = self.needed_per_attr()
        R = self.remaining_seats()
        per_attr_fail: Dict[str, float] = {}
        worst_attr = None
        worst_p = -1.0
        for attr, s_i in needed.items():
            if s_i <= 0:
                continue
            p_i = self.current_probability(attr)
            fail = self._attr_failure_prob(s_i, R, p_i)
            per_attr_fail[attr] = fail
            if fail > worst_p:
                worst_p = fail
                worst_attr = attr
        # Overall via union bound (lower bound on success)
        sum_fail = sum(per_attr_fail.values())
        overall_lb = max(0.0, 1.0 - sum_fail)
        return overall_lb, per_attr_fail, (worst_attr or "", worst_p if worst_p >= 0 else 0.0)

    def render_tui(self, last_msg: str) -> None:
        if not self.tui:
            return
        # Prepare data
        needed = self.needed_per_attr()
        R = self.remaining_seats()
        overall_lb, per_attr_fail, (tight_attr, tight_p) = self._estimate_success()
        # track history
        self.history_success.append(overall_lb)
        if len(self.history_success) > self.history_len:
            self.history_success = self.history_success[-self.history_len:]
        # Build screen
        lines = []
        lines.append(f"Game {self.game_id} | target={self.target} | accepted={self.accepted_total} | remaining={R}")
        lines.append(self._render_bar(self.accepted_total, self.target, width=50))
        lines.append(
            f"Policy={self.policy} k={self.cushion} alpha={self.alpha if self.policy=='chernoff' else 'n/a'} | updates={self.update_probs} (w={self.prior_weight})"
        )
        # Seat reserve
        total_need_line = sum(max(0, self.min_counts[a] - self.accepted_by_attr.get(a, 0)) for a in self.min_counts)
        seat_slack = R - total_need_line
        if self.use_seat_reserve:
            lines.append(f"Seat slack: {seat_slack} (reserve {self.reserve_slack})")
        else:
            lines.append(f"Seat slack: {seat_slack} (reserve disabled)")
        # Overall chance line
        label = "Success LB" if self.policy == "chernoff" else "Success est."
        lines.append(f"{label}: {overall_lb*100:.1f}% | tightest={tight_attr or '-'} p_fail={tight_p:.3g}")
        lines.append("Progress: " + self._render_sparkline(self.history_success, width=50))
        lines.append("")
        # Attribute rows: order by remaining need desc then name
        rows = [(a, needed[a]) for a in needed if needed[a] > 0]
        rows.sort(key=lambda kv: (-kv[1], kv[0]))
        if rows:
            lines.append("Quotas:")
            max_rows = self.tui_max_rows
            for a, s_i in rows[:max_rows]:
                have = self.accepted_by_attr.get(a, 0)
                need_total = self.min_counts.get(a, 0)
                bar = self._render_bar(have, need_total, width=40)
                p_i = self.current_probability(a)
                mu = R * p_i
                sigma = math.sqrt(mu * (1.0 - p_i)) if 0.0 < p_i < 1.0 else 0.0
                fail = per_attr_fail.get(a, 0.0)
                # seats required preview for current R
                k_eff = self.effective_k()
                n_req = self.seats_required_normal(s_i, p_i, k_eff, R)
                lines.append(
                    f"- {a:>16}: {bar} | need {s_i:>4} more | p={p_i:.3f} mu={mu:.1f} sig={sigma:.1f} fail≈{fail:.3g} | seats_req≈{n_req}"
                )
            extra = len(rows) - max_rows
            if extra > 0:
                lines.append(f"  (+{extra} more attributes hidden)")
        else:
            lines.append("All quotas satisfied.")
        lines.append("")
        lines.append(f"Last: {last_msg}")
        # Render: clear and home
        sys.stdout.write("\x1b[2J\x1b[H" + "\n".join(lines) + "\n")
        sys.stdout.flush()

    def remaining_seats(self) -> int:
        return self.target - self.accepted_total

    def needed_per_attr(self) -> Dict[str, int]:
        """Compute how many more positives are required for each attribute."""
        return {
            attr: max(0, self.min_counts[attr] - self.accepted_by_attr.get(attr, 0))
            for attr in self.min_counts
        }

    def should_accept(self, candidate_attrs: Set[str]) -> Tuple[bool, str]:
        """
        Decide whether to accept the current candidate.

        This function encapsulates the heart of the admission policy.  It
        returns ``(True, reason)`` if the candidate should be admitted and
        ``(False, reason)`` if the candidate should be rejected.
        """
        # How many seats remain?
        R = self.remaining_seats()
        if R <= 0:
            return False, "no seats remaining"

        # Outstanding needs per attribute
        needed = self.needed_per_attr()
        # If the candidate satisfies any outstanding need, accept them
        helpful = [attr for attr in candidate_attrs if needed.get(attr, 0) > 0]
        if helpful:
            return True, f"helpful: satisfies unmet {sorted(helpful)}"

        # Candidate does not help any constraints.  Compute whether we can
        # still meet every outstanding constraint if we admit them.
        new_R = R - 1
        # Hard feasibility guard based on seat reserve (optional)
        if self.use_seat_reserve:
            total_need = sum(v for v in needed.values())
            if new_R < total_need + self.reserve_slack:
                return False, (
                    f"unsafe: seat reserve short (new_R={new_R} < total_need={total_need} + slack={self.reserve_slack})"
                )
        # Per-attribute seats-budget guard (stronger, uses conservative p)
        if self.use_seats_budget_guard:
            k_eff_base = self.effective_k()
            sum_required = 0
            worst_attr = None
            worst_req = -1
            for attr, s_i in needed.items():
                if s_i <= 0:
                    continue
                p_i = self.current_probability(attr)
               # k renforcé si l'attribut est rare/tendu
                k_attr = k_eff_base * self.rarity_weight(p_i, s_i, new_R)
                n_req = self.seats_required_normal(s_i, p_i, k_attr, new_R)
                if n_req > worst_req:
                    worst_req = n_req
                    worst_attr = attr
                sum_required += n_req
                if sum_required > new_R:
                    return False, (
                        f"unsafe: seats budget short (sum_required={sum_required} > new_R={new_R}); "
                        f"tightest {worst_attr} needs {worst_req} seats"
                    )
            # Budget fits; accept
            return True, (
                f"safe: seats budget ok (sum_required={sum_required} <= new_R={new_R})"
            )
        # Estimate using a conservative lower bound on the number of
        # positives we will see for each attribute.  Using a normal
        # approximation to the binomial distribution, mu_i - k*sigma_i
        # gives us a k‑sigma lower bound on the number of successes.
        tight_attr = None
        tight_metric = None  # z-score for normal, prob for chernoff
        if self.policy == "fast":
            # Simple: require expected supply with cushion to cover needs
            # Use prior probabilities for speed
            for attr, s_i in needed.items():
                if s_i <= 0:
                    continue
                p_i = float(self.probabilities.get(attr, 0.0))
                mu = new_R * p_i
                if self.gamma * mu < s_i:
                    return False, (
                        f"unsafe: fast mu*c<{s_i} for {attr} (mu={mu:.1f}, c={self.gamma:.2f})"
                    )
            return True, "safe: fast expected supply ok"
        if self.policy == "chernoff":
            # Bonferroni split of alpha across unmet attributes
            unmet = [(a, s) for a, s in needed.items() if s > 0]
            m = max(1, len(unmet))
            # On alloue un alpha_i PLUS PETIT aux attributs rares (plus strict)
            inv_weights = []
            for a, s_i in unmet:
                p_i = self.current_probability(a)
                w = self.rarity_weight(p_i, s_i, new_R)  # w>=1 si rare/tendu
                inv_weights.append(1.0 / max(1e-9, w))
            sum_inv = sum(inv_weights) if inv_weights else 1.0
            worst_p = 0.0
            worst_attr = None
            for idx, (attr, s_i) in enumerate(unmet):
                p_i = self.current_probability(attr)
                if p_i <= 0.0:
                    return False, f"unsafe: need {attr} but p=0"
                mu = new_R * p_i
                if mu <= 0.0:
                    return False, f"unsafe: mu=0 for {attr}"
                # If need exceeds mean, the bound is trivially 1 (unsafe)
                if s_i >= mu:
                    return False, f"unsafe: need {attr}={s_i} >= mu={mu:.2f}"
                # alpha_i : plus rare => plus strict (alpha_i plus petit)
                per_attr_alpha = float(self.alpha) * (inv_weights[idx] / sum_inv)
                ratio = max(0.0, min(1.0, s_i / mu))
                delta = 1.0 - ratio  # in (0,1]
                # Lower-tail Chernoff: P[X < (1-d)mu] <= exp(-mu * d^2 / 2)
                prob_bound = math.exp(-mu * (delta ** 2) / 2.0)
                if prob_bound > worst_p:
                    worst_p = prob_bound
                    worst_attr = attr
                if prob_bound > per_attr_alpha:
                    return False, (
                        f"unsafe: chernoff P[X<{s_i}]>{per_attr_alpha:.3g} for {attr} "
                        f"(mu={mu:.2f}, bound={prob_bound:.3g})"
                    )
            tight_attr = worst_attr
            tight_metric = worst_p
            return True, (
                f"safe: chernoff ok; tightest {tight_attr} bound={tight_metric:.3g}"
            )
        else:
            # Normal k-sigma policy
            worst_z = float("inf")
            worst_attr = None
            for attr, s_i in needed.items():
                if s_i == 0:
                    continue  # Already satisfied
                p_i = self.current_probability(attr)
                if p_i <= 0.0:
                    # If the attribute never occurs, we must not use up seats
                    return False, f"unsafe: need {attr} but p=0"
                mu = new_R * p_i
                sigma = math.sqrt(mu * (1.0 - p_i)) if 0.0 < p_i < 1.0 else 0.0
                # k renforcé pour attribut rare/tendu
                k_attr = self.per_attr_k(p_i, s_i, new_R)
                lower_bound = mu - k_attr * sigma
                # If the lower bound falls short of the required count, reject
                if lower_bound < s_i:
                    return False, (
                        f"unsafe: normal LB<{s_i} for {attr} "
                        f"(mu={mu:.2f}, sigma={sigma:.2f}, k={k_attr:.2f}, LB={lower_bound:.2f})"
                    )
                # Track tightness via z = (mu - s) / sigma (higher is safer)
                if sigma > 0:
                    z = (mu - s_i) / sigma
                else:
                    z = float("inf") if mu >= s_i else -float("inf")
                if z < worst_z:
                    worst_z = z
                    worst_attr = attr
            tight_attr = worst_attr
            tight_metric = worst_z
            # All constraints can likely still be met; admit the candidate
            return True, (
                f"safe: normal ok; tightest {tight_attr} z={tight_metric:.2f}"
            )

    def update_after_decision(self, candidate_attrs: Set[str], accepted: bool) -> None:
        """Update internal state after an accept/reject decision."""
        if accepted:
            self.accepted_total += 1
            for attr in candidate_attrs:
                if attr in self.accepted_by_attr:
                    self.accepted_by_attr[attr] += 1
        else:
            self.rejected_count += 1

    def observe_candidate(self, candidate_attrs: Set[str]) -> None:
        """Record the attributes of the candidate we've just observed."""
        self.seen_count += 1
        for attr in candidate_attrs:
            if attr in self.seen_by_attr:
                self.seen_by_attr[attr] += 1
            else:
                self.seen_by_attr[attr] = 1


def create_game(base_url: str, player_id: str, scenario: int) -> Dict:
    """
    Start a new game on the server.

    Returns the JSON response from the ``/new-game`` endpoint.  Raises
    ``requests.HTTPError`` if the request fails.
    """
    resp = requests.get(
        f"{base_url}/new-game",
        params={"scenario": scenario, "playerId": player_id},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def decide_and_next(base_url: str, game_id: str, person_index: int, accept: bool | None) -> Dict:
    """
    Query the next person and optionally supply a decision for the previous one.

    ``accept`` may be ``True`` or ``False`` to indicate that the last
    person was admitted or rejected, or ``None`` for the very first call.
    The server returns a JSON object containing the current status,
    updated counts, and the next person (if the game is running).
    """
    params = {"gameId": game_id, "personIndex": person_index}
    if accept is not None:
        # Accept must be supplied as a lower‑case string, not a boolean
        params["accept"] = "true" if accept else "false"
    resp = requests.get(f"{base_url}/decide-and-next", params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_solver(
    base_url: str,
    player_id: str,
    scenario: int,
    target: int,
    k: float,
    *,
    policy: str = "normal",
    alpha: float = 0.01,
    update_probs: bool = False,
    prior_weight: float = 200.0,
    verbose: bool = False,
    log_every: int = 10,
    reserve_slack: int = 0,
    use_seat_reserve: bool = True,
    use_seats_budget_guard: bool = True,
    gamma: float = 0.85,
    tui: bool = True,
    tui_every: int = 5,
    tui_max_rows: int = 12,
) -> int:
    """
    Run the solver for a single scenario and return the number of rejections.

    This function orchestrates the entire game: it creates a new game,
    repeatedly fetches candidates and makes decisions, and stops when the
    server reports completion or failure.  The final rejection count
    reported by the server is returned.  If the server fails the game
    before completion, this function raises ``RuntimeError``.
    """
    init_data = create_game(base_url, player_id, scenario)
    game_id = init_data["gameId"]
    # Extract quotas and probabilities
    min_counts: Dict[str, int] = {}
    for constraint in init_data.get("constraints", []):
        attr = str(constraint["attribute"])
        min_counts[attr] = int(constraint["minCount"])
    relative_freqs = init_data.get("attributeStatistics", {}).get("relativeFrequencies", {})
    probabilities: Dict[str, float] = {}
    for attr, freq in relative_freqs.items():
        probabilities[str(attr)] = float(freq)
    # Create game state
    state = GameState(
        game_id=game_id,
        target=target,
        probabilities=probabilities,
        min_counts=min_counts,
        cushion=k,
        policy=policy,
        alpha=alpha,
        update_probs=update_probs,
        prior_weight=prior_weight,
        verbose=verbose,
        log_every=log_every,
        reserve_slack=reserve_slack,
        use_seat_reserve=use_seat_reserve,
        use_seats_budget_guard=use_seats_budget_guard,
        gamma=gamma,
        tui=tui,
        tui_every=tui_every,
        tui_max_rows=tui_max_rows,
    )
    if verbose and not state.tui:
        print(
            f"Game {game_id} started: scenario={scenario}, target={target}, policy={policy}, "
            f"k={k}, alpha={alpha if policy=='chernoff' else 'n/a'}, update_probs={update_probs}, "
            f"prior_weight={prior_weight}, log_every={log_every}"
        )
        print(f"Quotas (min_counts): { {k: v for k, v in sorted(min_counts.items())} }")
        print(
            f"Attribute priors: { {k: round(v, 4) for k, v in sorted(probabilities.items())} }"
        )
    else:
        # Initial TUI render
        state.render_tui("initialized")
    accept: bool | None = None
    # Main loop
    while True:
        response = decide_and_next(base_url, game_id, state.person_index, accept)
        status = response.get("status", "")
        if status == "running":
            # Obtain next person and decide
            next_person = response.get("nextPerson", {})
            # Attributes are provided as {attributeId: boolean}
            raw_attrs = next_person.get("attributes", {})
            candidate_attrs = {str(attr) for attr, val in raw_attrs.items() if val}
            # Observe the candidate for online probability updates
            state.observe_candidate(candidate_attrs)
            # Pre-decision snapshot
            R = state.remaining_seats()
            needs_snapshot = {a: n for a, n in state.needed_per_attr().items() if n > 0}
            # Use policy to decide acceptance
            accepted, reason = state.should_accept(candidate_attrs)
            # Logging
            if state.verbose and not state.tui:
                idx = int(next_person.get("personIndex", state.person_index))
                do_log = state.log_every <= 1 or (idx % state.log_every == 0) or (not accepted) or reason.startswith("helpful")
                if do_log:
                    # Keep needs summary short
                    needs_items = sorted(needs_snapshot.items(), key=lambda kv: (-kv[1], kv[0]))
                    needs_short = dict(needs_items[:6])
                    print(
                        f"idx={idx} R={R} cand={sorted(candidate_attrs)} => "
                        f"{'ACCEPT' if accepted else 'REJECT'}; {reason}; needs={needs_short}"
                    )
            # Update state
            state.update_after_decision(candidate_attrs, accepted)
            # TUI update
            if state.tui:
                idx = int(next_person.get("personIndex", state.person_index))
                do_draw = state.tui_every <= 1 or (idx % state.tui_every == 0) or (not accepted) or reason.startswith("helpful")
                if do_draw:
                    last_msg = (
                        f"idx={idx} {'ACCEPT' if accepted else 'REJECT'}; {reason}; cand={sorted(candidate_attrs)}"
                    )
                    state.render_tui(last_msg)
            # Set up for next call
            accept = accepted
            state.person_index = int(next_person.get("personIndex", 0))
        elif status == "completed":
            # Finished successfully
            rejected = int(response.get("rejectedCount", state.rejected_count))
            # Final render with completion line
            if state.tui:
                state.render_tui(
                    f"completed: accepted={state.accepted_total}, rejected={rejected}"
                )
            else:
                print(
                    f"Scenario {scenario} completed: {state.accepted_total} accepted, {rejected} rejected."
                )
            return rejected
        else:
            reason = response.get("reason", "unknown")
            raise RuntimeError(f"Game failed: {reason}")


def auto_optimize(
    base_url: str,
    player_id: str,
    scenario: int,
    target: int,
    *,
    max_games: int = 12,
    goal_rejections: Optional[int] = None,
    # Baseline knobs (used if not overridden by a grid entry)
    policy: str = "chernoff",
    alpha: float = 0.002,
    k: float = 2.5,
    update_probs: bool = True,
    prior_weight: float = 300.0,
    reserve_slack: int = 0,
    verbose: bool = False,
    log_every: int = 10,
    loose: bool = False,
    use_seat_reserve: bool = True,
    use_seats_budget_guard: bool = True,
    gamma: float = 0.85,
) -> None:
    """Run multiple games and try to improve the score by varying parameters.

    This cycles over a conservative grid of parameter configurations and tracks
    the best (lowest) rejection count. Stops early if `goal_rejections` is met.
    """
    # Parameter grid to explore
    grid: List[Dict[str, object]] = []
    # Chernoff-focused conservative set
    for a in [0.002, 0.001, 0.0007, 0.0005]:
        for pw in [300.0, 500.0]:
            for rs in [0, 10, 20, 30]:
                grid.append({"policy": "chernoff", "alpha": a, "prior_weight": pw, "reserve_slack": rs, "use_seat_reserve": True})
    # Normal policy variants
    for kk in [3.6, 3.3, 3.0]:
        for pw in [300.0, 500.0]:
            for rs in [0, 10, 20]:
                grid.append({"policy": "normal", "k": kk, "prior_weight": pw, "reserve_slack": rs, "use_seat_reserve": True})
    # Fast policy variants (simple, high-throughput)
    for g in [0.75, 0.8, 0.85, 0.9]:
        for rs in [0, 10]:
            grid.append({"policy": "fast", "gamma": g, "reserve_slack": rs, "use_seat_reserve": False})
    # Loose/aggressive expansions if requested
    if loose:
        # Higher alpha, lower k, lower prior weight, negative reserve slack, and disabling reserve
        for a in [0.005, 0.01, 0.02, 0.05]:
            for pw in [100.0, 150.0, 200.0]:
                for rs in [-20, -10, 0]:
                    grid.append({"policy": "chernoff", "alpha": a, "prior_weight": pw, "reserve_slack": rs, "use_seat_reserve": True})
                    grid.append({"policy": "chernoff", "alpha": a, "prior_weight": pw, "reserve_slack": rs, "use_seat_reserve": False})
        for kk in [2.5, 2.2, 2.0]:
            for pw in [100.0, 150.0, 200.0]:
                for rs in [-20, -10, 0]:
                    grid.append({"policy": "normal", "k": kk, "prior_weight": pw, "reserve_slack": rs, "use_seat_reserve": True})
                    grid.append({"policy": "normal", "k": kk, "prior_weight": pw, "reserve_slack": rs, "use_seat_reserve": False})

    best_rej: Optional[int] = None
    best_cfg: Optional[Dict[str, object]] = None
    attempts = 0
    print(f"Auto optimize: scenario={scenario}, max_games={max_games}, goal={goal_rejections if goal_rejections is not None else '-'}")

    while attempts < max_games:
        cfg = grid[attempts % len(grid)].copy()
        attempts += 1
        # Fill defaults
        cfg.setdefault("policy", policy)
        cfg.setdefault("alpha", alpha)
        cfg.setdefault("k", k)
        cfg.setdefault("prior_weight", prior_weight)
        cfg.setdefault("reserve_slack", reserve_slack)
        cfg.setdefault("gamma", gamma)
        cfg.update({"update_probs": update_probs})

        # Prepare run args
        run_kwargs = dict(
            base_url=base_url,
            player_id=player_id,
            scenario=scenario,
            target=target,
            k=float(cfg.get("k", k)),
            policy=str(cfg.get("policy", policy)),
            alpha=float(cfg.get("alpha", alpha)),
            update_probs=bool(cfg.get("update_probs", update_probs)),
            prior_weight=float(cfg.get("prior_weight", prior_weight)),
            verbose=verbose,
            log_every=log_every,
            reserve_slack=int(cfg.get("reserve_slack", reserve_slack)),
            use_seat_reserve=bool(cfg.get("use_seat_reserve", use_seat_reserve)),
            gamma=float(cfg.get("gamma", gamma)),
            use_seats_budget_guard=bool(cfg.get("use_seats_budget_guard", use_seats_budget_guard)),
            tui=False,  # disable TUI during auto optimization
        )

        print(
            "Attempt {idx}/{max}: policy={policy} k={k} alpha={alpha} gamma={gamma} update={upd} w={pw} reserve={rs}...".format(
                idx=attempts,
                max=max_games,
                policy=run_kwargs["policy"],
                k=run_kwargs["k"],
                alpha=run_kwargs["alpha"],
                gamma=run_kwargs["gamma"],
                upd=run_kwargs["update_probs"],
                pw=run_kwargs["prior_weight"],
                rs=run_kwargs["reserve_slack"],
            )
        )
        try:
            rej = run_solver(**run_kwargs)
            ok = True
        except Exception as exc:
            rej = 1_000_000  # treat failure as terrible
            ok = False
            print(f"  Result: FAILED ({exc})")

        if ok:
            print(f"  Result: completed with {rej} rejections")
        # Track best
        if best_rej is None or rej < best_rej:
            best_rej = rej
            best_cfg = run_kwargs
            print("  New best!")
            # Add local refinements near the best configuration
            try:
                if best_cfg["policy"] == "chernoff":
                    new_alpha = max(0.0002, float(best_cfg["alpha"]) * 0.7)
                    grid.append({
                        "policy": "chernoff",
                        "alpha": new_alpha,
                        "prior_weight": float(best_cfg["prior_weight"]),
                        "reserve_slack": int(best_cfg["reserve_slack"]),
                        "use_seat_reserve": bool(best_cfg.get("use_seat_reserve", True)),
                    })
                    grid.append({
                        "policy": "chernoff",
                        "alpha": float(best_cfg["alpha"]),
                        "prior_weight": float(best_cfg["prior_weight"]),
                        "reserve_slack": int(best_cfg["reserve_slack"]) + 10,
                        "use_seat_reserve": bool(best_cfg.get("use_seat_reserve", True)),
                    })
                else:  # normal
                    new_k = min(4.0, float(best_cfg["k"]) + 0.3)
                    grid.append({
                        "policy": "normal",
                        "k": new_k,
                        "prior_weight": float(best_cfg["prior_weight"]),
                        "reserve_slack": int(best_cfg["reserve_slack"]),
                        "use_seat_reserve": bool(best_cfg.get("use_seat_reserve", True)),
                    })
                    grid.append({
                        "policy": "normal",
                        "k": float(best_cfg["k"]),
                        "prior_weight": float(best_cfg["prior_weight"]),
                        "reserve_slack": int(best_cfg["reserve_slack"]) + 10,
                        "use_seat_reserve": bool(best_cfg.get("use_seat_reserve", True)),
                    })
            except Exception:
                pass
        # Stop if goal met
        if goal_rejections is not None and best_rej is not None and best_rej <= goal_rejections:
            break

    print("Optimization finished.")
    if best_rej is None:
        print("No successful runs.")
        return
    # Summarize best and suggest a command
    print(
        "Best: {rej} rejections | policy={pol} k={k} alpha={a} update={upd} w={pw} reserve={rs}".format(
            rej=best_rej,
            pol=best_cfg["policy"],
            k=best_cfg["k"],
            a=best_cfg["alpha"],
            upd=best_cfg["update_probs"],
            pw=best_cfg["prior_weight"],
            rs=best_cfg["reserve_slack"],
        )
    )
    print("Re-run with: ")
    print(
        "python berghain_solver.py --base-url {bu} --player-id {pid} --scenario {sc} --policy {pol} --k {k} --alpha {a} --prior-weight {pw} --reserve-slack {rs} {upd}".format(
            bu=base_url,
            pid=player_id,
            sc=scenario,
            pol=best_cfg["policy"],
            k=best_cfg["k"],
            a=best_cfg["alpha"],
            pw=best_cfg["prior_weight"],
            rs=best_cfg["reserve_slack"],
            upd="--update-probabilities" if best_cfg["update_probs"] else "",
        )
    )


def main(args: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Solve the Listen Labs Berghain challenge.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BERGHAIN_BASE_URL"),
        help=(
            "Base URL of the challenge endpoints (e.g. https://example.com). "
            "Defaults from env BERGHAIN_BASE_URL if not provided."
        ),
    )
    parser.add_argument(
        "--player-id",
        default=os.environ.get("BERGHAIN_PLAYER_ID"),
        help=(
            "Your player UUID (from the challenge website). "
            "Defaults from env BERGHAIN_PLAYER_ID if not provided."
        ),
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Scenario number to play (1, 2, or 3)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1000,
        help="Number of people to admit to fill the venue (default 1000)",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=2.5,
        help=(
            "Number of standard deviations to demand as cushion when admitting"
            " unhelpful candidates; higher values make the solver more conservative."
        ),
    )
    parser.add_argument(
        "--policy",
        choices=["normal", "chernoff", "fast"],
        default="chernoff",
        help="Decision guard policy: 'fast' (mu*cushion), 'normal' (k-sigma), or 'chernoff'",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.002,
        help="Overall failure budget alpha for chernoff policy (split across unmet attributes)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.85,
        help="Fast policy cushion: require gamma * expected supply >= need (0<gamma<=1)",
    )
    parser.add_argument(
        "--update-probabilities",
        action="store_true",
        default=True,
        help="Update attribute probabilities online by blending observed frequencies with priors",
    )
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=300.0,
        help="Pseudo-count weight for prior when blending probabilities (higher = slower adaptation)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print step-by-step decisions and diagnostics",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="When verbose, print every N candidates (default 10; use 1 for every step)",
    )
    parser.add_argument(
        "--reserve-slack",
        type=int,
        default=0,
        help=(
            "Seat reserve: require remaining seats to be at least the sum of outstanding "
            "needs plus this slack before accepting unhelpful candidates (can be negative)"
        ),
    )
    parser.add_argument(
        "--no-seat-reserve",
        dest="use_seat_reserve",
        action="store_false",
        help="Disable seat reserve guard entirely (looser)",
    )
    parser.set_defaults(use_seat_reserve=True)
    parser.add_argument(
        "--no-seats-budget-guard",
        dest="use_seats_budget_guard",
        action="store_false",
        help="Disable per-attribute seats-budget guard (not recommended)",
    )
    parser.set_defaults(use_seats_budget_guard=True)
    parser.add_argument(
        "--tui",
        dest="tui",
        action="store_true",
        default=True,
        help="Enable in-terminal dashboard with progress bars and success estimate",
    )
    parser.add_argument(
        "--no-tui",
        dest="tui",
        action="store_false",
        help="Disable in-terminal dashboard",
    )
    parser.add_argument(
        "--tui-every",
        type=int,
        default=5,
        help="Refresh dashboard every N candidates (default 5; use 1 for every step)",
    )
    parser.add_argument(
        "--tui-max-rows",
        type=int,
        default=12,
        help="Max quota rows to show in dashboard (default 12)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-optimization: loop new games, vary parameters, and seek a better score",
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="Use a looser/aggressive parameter grid in auto mode (higher alpha, lower k, lower prior weight, negative/disabled seat reserve)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=12,
        help="Maximum number of games to run in auto mode (default 12)",
    )
    parser.add_argument(
        "--goal-rejections",
        type=int,
        default=None,
        help="Stop auto mode when the best rejection count is <= this value",
    )
    parsed = parser.parse_args(args)
    # Validate required inputs (support env fallbacks)
    if not parsed.base_url:
        print(
            "Error: --base-url is required (or set BERGHAIN_BASE_URL).",
            file=sys.stderr,
        )
        sys.exit(2)
    if not parsed.player_id:
        print(
            "Error: --player-id is required (or set BERGHAIN_PLAYER_ID).",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        if parsed.auto:
            auto_optimize(
                base_url=parsed.base_url,
                player_id=parsed.player_id,
                scenario=parsed.scenario,
                target=parsed.target,
                max_games=parsed.max_games,
                goal_rejections=parsed.goal_rejections,
                policy=parsed.policy,
                alpha=parsed.alpha,
                k=parsed.k,
                gamma=parsed.gamma,
                update_probs=parsed.update_probabilities,
                prior_weight=parsed.prior_weight,
                reserve_slack=parsed.reserve_slack,
                use_seat_reserve=parsed.use_seat_reserve,
                use_seats_budget_guard=parsed.use_seats_budget_guard,
                verbose=parsed.verbose,
                log_every=parsed.log_every,
                loose=parsed.loose,
            )
        else:
            run_solver(
                base_url=parsed.base_url,
                player_id=parsed.player_id,
                scenario=parsed.scenario,
                target=parsed.target,
                k=parsed.k,
                policy=parsed.policy,
                alpha=parsed.alpha,
                gamma=parsed.gamma,
                update_probs=parsed.update_probabilities,
                prior_weight=parsed.prior_weight,
                verbose=parsed.verbose,
                log_every=parsed.log_every,
                reserve_slack=parsed.reserve_slack,
                use_seat_reserve=parsed.use_seat_reserve,
                use_seats_budget_guard=parsed.use_seats_budget_guard,
                tui=parsed.tui,
                tui_every=parsed.tui_every,
                tui_max_rows=parsed.tui_max_rows,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
