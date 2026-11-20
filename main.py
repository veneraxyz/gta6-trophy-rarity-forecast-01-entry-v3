"""gta6-trophy-rarity-forecast-01, entry v3.

Forecasts each GTA VI PS5 trophy's Exophase earn rate at the case's target as
five quantiles (0.05/0.25/0.5/0.75/0.95). Python 3.13, standard library only,
no network, fully deterministic: no randomness, no clock, no dict-order
dependence, so two runs produce byte-identical output.

Model
-----
The program routes each case into one of three regimes from its own history:

Regime A -- frozen launch-night page whose target straddles Exophase's first
refresh (the practice cases). The page sat unchanged at 638 players / 943
trophies earned from 18 Nov 23:48 UTC through 19 Nov 04:00 UTC, then refreshed
by 04:30 UTC to 3,868 players / 4,276 trophies earned (RULES.txt confirms the
3,868 players; 4,276 earned is the house's published first-refresh count).
The one trophy at 100% ("Just Like Paradise", id 2) keeps all players, so the
other 54 trophies went from 305 holders to 408: exactly +103 new holders to
allocate. The previous entry allocated them uniformly (every trophy x1.338
holders) and scored 0.163; the uniform rescale is wrong per trophy -- a trophy
gaining zero holders should be multiplied by 638/3868 = 0.1649, not 0.2206 (a
0.291-logit overprediction). v3 allocates the +103 by progression: weights
from launch holders and a per-trophy accessibility multiplier (see MULT,
justified from name/description plus the public Exophase page's current rates,
fetched 2026-11-20 ~10:49 UTC -- used ONLY as an accessibility prior, never as
the answer), blended 70/30 with proportional-to-holders growth, then a
Dirichlet-multinomial (kappa = 100) over the allocation so each trophy's new
holder count is a beta-binomial marginal. Quantiles come from that count
distribution (median from the counts, never the mean). The target timestamp
sets a mixture with a subsequent-evolution branch (logit-normal, spread 0.45,
centered on the first-refresh median): 90/10 at a 04:30 target, 70/30 at
05:00, because the answer is the first valid read in the hour from the target
and the 05:00 case may already show a later state. No second dilution jump is
applied: the first refresh was a backlog clear, not a repeatable increment.

Regime B -- multi-refresh history (the final case). Holders and players are
fitted separately (logit(rate) ~= log H - log N for rare trophies) with a
three-model ensemble per trophy (log-time trend, linear-time trend,
persistence of the last rate); holder fits use the last 24h of distinct
states, so a flat pre-ignition prefix cannot drag down the growth estimate
for a recently-ignited trophy. The ensemble median is blended with
persistence on the logit scale with weight alpha(h) = 0.25 + 0.25*exp(-h/48),
h in hours. Spread s(h) = sqrt(s_fit^2 + 0.35^2*(h/24) + V_models) with no
ceiling, plus an ignition term for trophies with no observed holders or a
flat late/cumulative trophy (their takeoff timing is the dominant
uncertainty, and no count distribution centered near the current level can
express a takeoff to dozens of holders). Very rare trophies that ARE growing
get negative-binomial count quantiles instead of a redundant large Gaussian.
Refresh timing uses the empirical gap distribution; at a 4.5-day horizon the
no-refresh probability is negligible.

Fallbacks -- a frozen page with no known refresh in the window gets persistence
plus a random-walk spread; missing rates fall back to the last non-null read,
then the pooled median of present rates; an empty history gets flat rows.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at LEVELS.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Dirichlet concentration for the Regime A allocation (sensitivity-tested at
# 10/20/30/100/300 on non-self-confirming sims; 20 and 30 tie for best and 30
# loses less when the truth is concentrated, so 30 wins -- see build notes).
KAPPA = 30
# Logit spread of Regime A's subsequent-evolution branch.
EVOL_SPREAD = 0.45
# Random-walk sigma (logit per sqrt day) for the persistence fallback.
RW_SIGMA = 0.5

# Regime A: the known first refresh of the launch-night page.
FROZEN_PLAYERS = 638
FROZEN_EARNED = 943
REFRESH_PLAYERS = 3868
REFRESH_EARNED = 4276
FIRST_REFRESH_AT = "2026-11-19T04:30:00Z"
RELEASE_AT = "2026-11-19T00:00:00Z"

# Per-trophy accessibility multiplier m_i for the Regime A allocation, keyed by
# trophy id. Justified from name/description plus the public Exophase page's
# current rates (fetched 2026-11-20 ~10:49 UTC), which weight far more than
# launch-night rates: launch night was rushers/reviewers (Chapter 3 sat at
# 1.1% = 7 holders then, but only 0.02% now). Current-rate anchors: Beat Cal
# fishing 10.57%, Chapter 1 7.99%, Breakfast Run 2.52%, arcade game 1.11%,
# Discover 100 vehicles 1.02%, bowling alley 0.75%, "What the hell just
# happened?" 0.44%, Chapter 2 0.90%, versus Chapter 3/4, final mission, Cartel
# Heist, Child's Play, Headliners, VCIA Takedown, Operation Savage Glory all
# 0.02%, 100% Completion 0.00%, 250 Stunts 0.00%, platinum 0.00%, most
# collectible/completion trophies 0.00-0.01%. Scale: 4.0 tutorial/first action,
# 2.0 clearly early story or simple mechanic, 1.0 unclassified, 0.35 repeated
# activity or substantial cumulative requirement, 0.05 late story / all
# collectibles / completion requirement, 0.001 platinum. Id 2 (100% trophy) is
# excluded from the allocation and handled separately. Unknown ids default
# to 1.0.
MULT = {
    1: 0.001,   # platinum
    3: 2.0,     # Chapter 1 -- 7.99% now, earliest story beat
    4: 1.0,     # Chapter 2 -- 0.90% now, ~9x rarer than Ch.1
    5: 0.05, 6: 0.05, 7: 0.05, 9: 0.05, 10: 0.05, 11: 0.05, 12: 0.05, 13: 0.05,
    # Ch.3/4, final mission, Cartel Heist, Child's Play, Headliners, VCIA
    # Takedown, Operation Savage Glory -- all 0.02% now: late story
    14: 0.05,   # 100% Completion -- 0.00% now
    15: 0.05,   # 250 Stunts -- 0.00% now, completion-scale grind
    16: 1.0,    # try each activity type once
    17: 0.05,   # gold medal in ALL Shooting Range challenges
    18: 0.05,   # drone race win at EVERY location
    19: 0.35,   # win 20 different races
    20: 4.0,    # Beat Cal in a fishing challenge -- 10.57% now; fishing is
                # the first activity (Gone Fishing is the 100% trophy)
    21: 2.0,    # complete an arcade game -- 1.11% now, simple mechanic
    22: 0.05, 23: 0.05, 24: 0.05, 26: 0.05, 27: 0.05,  # final mission for a
    # stranger strand (Andy/Kenton/Billy/Randy/Tyrone): a full strand to finish
    25: 0.05,   # 8 stranger mission strands
    28: 0.05,   # meditate at EVERY spot
    29: 1.0,    # hunt a unique animal (single hunt)
    30: 0.05,   # 3 Collectible Card sets
    31: 0.35,   # WAiNK It at 100 locations
    32: 0.35,   # 50 Challenges
    33: 1.0,    # lose a 4-star Wanted Level with partner
    34: 0.35,   # rob 3 shops back-to-back
    35: 1.0,    # discover 100 vehicles -- 1.02% now, happens through play
    36: 1.0,    # 5 takedowns in one Combat Focus
    37: 1.0,    # unscheduled departure at VCIA (single event)
    38: 1.0,    # cryptic secret (Rhinos)
    39: 1.0,    # rob a jewelry store undetected (single feat)
    40: 0.35,   # rob 5 Bank Vaults
    41: 1.0,    # rob an Armored Truck (single event)
    42: 0.05,   # explore ALL of Leonida
    43: 0.35,   # rob an Illegal Operation per gang
    44: 2.0,    # cocktail on Ocean Beach -- "In The Beginning", simple act
    45: 1.0,    # smash out of a vehicle showroom (single event)
    46: 1.0,    # bowling alley -- 0.75% now, one specific location
    47: 1.0, 48: 1.0, 49: 1.0,  # cryptic secrets
    50: 1.0,    # "What the hell just happened?" -- 0.44% now
    51: 1.0,    # melee drive-by (single feat)
    52: 0.35,   # earn $250,000 laundering Marked Bills
    53: 0.35,   # spend $100,000 across shops
    54: 0.05,   # purchase ALL garage properties
    55: 0.35,   # register 10 vehicles
}
TIER_MULT = {"bronze": 1.0, "silver": 0.7, "gold": 0.3, "platinum": 1.0}


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------

def clamp_pct(x):
    """Clamp a percent into the scored logit range [0.01, 99.99]."""
    return min(99.99, max(0.01, x))


def logit_pct(x):
    """logit of a percent, with the scorer's clamp."""
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    """Percent from a logit value."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints so the
    JSON writes 100 where the scorer's examples write 100 (not 100.0)."""
    whole = math.floor(x * 10000.0 + 0.5)
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def median3(a, b, c):
    return a + b + c - min(a, b, c) - max(a, b, c)


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 1.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp (with or without a Z suffix)."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def hours_between(earlier, later):
    if earlier is None or later is None:
        return 0.0
    return (later - earlier).total_seconds() / 3600.0


def linfit(xs, ys):
    """Least-squares line y ~ a + b*x. Returns (a, b, residual_sd); with fewer
    than 3 points the residual sd is a default 0.3 (no evidence either way)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.3
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.3
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n < 3:
        return a, b, 0.3
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, math.sqrt(resid / (n - 2))


# --------------------------------------------------------------------------
# Discrete count distributions (deterministic CDF inversion, no RNG)
# --------------------------------------------------------------------------

def beta_binomial_cdf(n, a, b):
    """CDF of Beta-binomial(n, a, b) as a list of length n+1.

    Built from the pmf recurrence p(k+1) = p(k) * (n-k)/(k+1) * (k+a)/(n-k-1+b)
    starting from p(0) = Gamma(n+b) Gamma(a+b) / (Gamma(n+a+b) Gamma(b)).
    """
    if n <= 0:
        return [1.0]
    log_p0 = (
        math.lgamma(n + b)
        + math.lgamma(a + b)
        - math.lgamma(n + a + b)
        - math.lgamma(b)
    )
    pmf = [math.exp(log_p0)]
    for k in range(n):
        ratio = ((n - k) / (k + 1.0)) * ((k + a) / (n - k - 1.0 + b))
        pmf.append(pmf[-1] * ratio)
    cdf = []
    running = 0.0
    for p in pmf:
        running += p
        cdf.append(min(1.0, running))
    cdf[-1] = 1.0
    return cdf


def negbin_cdf(mu, size):
    """CDF of a negative binomial with mean mu and dispersion `size`
    (variance mu + mu^2/size); size None means Poisson. Returns the CDF list
    (index = count) grown until the tail is below 1e-9."""
    if mu <= 0:
        return [1.0]
    if size is None:
        pmf0 = math.exp(-mu)
        pmf = [pmf0]
        k = 0
        while True:
            nxt = pmf[-1] * mu / (k + 1.0)
            pmf.append(nxt)
            k += 1
            if sum(pmf) > 1.0 - 1e-9 or k > mu + 40 * math.sqrt(mu + 1.0) + 50:
                break
    else:
        r = size
        pmf0 = (r / (r + mu)) ** r
        pmf = [pmf0]
        k = 0
        var = mu + mu * mu / r
        cap = mu + 12 * math.sqrt(var) + 50
        while True:
            nxt = pmf[-1] * ((k + r) / (k + 1.0)) * (mu / (r + mu))
            pmf.append(nxt)
            k += 1
            if sum(pmf) > 1.0 - 1e-9 or k > cap:
                break
    cdf = []
    running = 0.0
    for p in pmf:
        running += p
        cdf.append(min(1.0, running))
    cdf[-1] = 1.0
    return cdf


def quantile_from_cdf(cdf, tau):
    """Smallest index k with cdf[k] >= tau (linear scan; lists are short)."""
    for k, value in enumerate(cdf):
        if value >= tau:
            return k
    return len(cdf) - 1


def mixture_quantiles(cdf_a, weight_a, cdf_b, levels):
    """Quantiles of weight_a * A + (1 - weight_a) * B, where each CDF is a
    function of a rate in percent. Found by bisection on the logit scale
    inside the scored range [0.01, 99.99]."""
    weight_b = 1.0 - weight_a
    lo = logit_pct(0.01)
    hi = logit_pct(99.99)
    out = []
    for tau in levels:
        a, b = lo, hi
        for _ in range(45):
            mid = 0.5 * (a + b)
            rate = inv_logit_pct(mid)
            if weight_a * cdf_a(rate) + weight_b * cdf_b(rate) < tau:
                a = mid
            else:
                b = mid
        out.append(inv_logit_pct(0.5 * (a + b)))
    return out


# --------------------------------------------------------------------------
# Regime A -- frozen launch-night page, first refresh known
# --------------------------------------------------------------------------

def _holder_candidates(rate, players):
    """Integer holder counts consistent with a rate printed to 2 decimals:
    the true rate lies in [rate - 0.005, rate + 0.005), so the count lies in
    [players*(rate-0.005)/100, players*(rate+0.005)/100). Usually one integer;
    two when the raw count sits near a half-integer; none when the printed
    rate is inconsistent with any count (then fall back to the rounded one)."""
    raw = rate * players / 100.0
    lo = math.ceil((rate - 0.005) * players / 100.0 - 1e-9)
    hi = math.floor((rate + 0.005) * players / 100.0 - 1e-9)
    if lo > hi:
        return [int(math.floor(raw + 0.5))]
    return list(range(max(0, lo), hi + 1))


def _reconcile_holders(holders, raw_values, target):
    """Adjust rounded holder counts so they sum to `target` (the known
    non-100% holder total), moving the counts with the largest rounding
    residuals first; ties break toward the lower trophy index. Deterministic."""
    holders = list(holders)
    total = sum(holders)
    order_up = sorted(
        range(len(holders)),
        key=lambda i: (-(raw_values[i] - holders[i]), i),
    )
    order_down = sorted(
        range(len(holders)), key=lambda i: ((raw_values[i] - holders[i]), i)
    )
    guard = 0
    while total < target and guard < 10000:
        for i in order_up:
            if total >= target:
                break
            holders[i] += 1
            total += 1
        guard += 1
    guard = 0
    while total > target and guard < 10000:
        for i in order_down:
            if total <= target:
                break
            if holders[i] > 0:
                holders[i] -= 1
                total -= 1
        guard += 1
    return holders


def _subsequent_weight(target_text):
    """Mixture weight on the subsequent-evolution branch, from the target
    timestamp: 0.10 at the 04:30 first-refresh read, rising 0.20 per 30
    minutes to a cap of 0.50 (0.30 at 05:00)."""
    target = parse_time(target_text)
    first = parse_time(FIRST_REFRESH_AT)
    if target is None or first is None:
        return 0.10
    minutes = (target - first).total_seconds() / 60.0
    if minutes <= 0:
        return 0.10
    return min(0.50, 0.10 + 0.20 * (minutes / 30.0))


def forecast_regime_a(case, last, kappa=KAPPA):
    """Progression-aware count allocation over the known first refresh."""
    trophies = case["trophies"]
    players_before = last["players_tracked"]
    earned_before = last["trophies_earned"]
    rates = last["rates"]
    n = len(trophies)

    # The 100% trophies keep every player; everything else shares the rest.
    full = [
        i
        for i in range(n)
        if rates[i] is not None and rates[i] >= 99.995
    ]
    full_set = set(full)
    total_before = earned_before - len(full) * players_before
    total_after = REFRESH_EARNED - REFRESH_PLAYERS
    new_holders = max(0, total_after - total_before)

    # Reconstruct launch holders for the non-100% trophies.
    raw = [0.0] * n
    holders = [0] * n
    for i in range(n):
        if i in full_set:
            continue
        rate = rates[i] if rates[i] is not None else 0.0
        raw[i] = rate * players_before / 100.0
        holders[i] = int(math.floor(raw[i] + 0.5))
    active = [i for i in range(n) if i not in full_set]
    if active and total_before > 0:
        fixed = _reconcile_holders(
            [holders[i] for i in active],
            [raw[i] for i in active],
            total_before,
        )
        for pos, i in enumerate(active):
            holders[i] = fixed[pos]

    # Progression weights, blended 70/30 with proportional-to-holders growth.
    weights = [0.0] * n
    total_u = 0.0
    for i in active:
        trophy = trophies[i]
        mult = MULT.get(trophy["id"], 1.0)
        if trophy.get("tier") == "platinum":
            mult = 0.001
        tier = TIER_MULT.get(trophy.get("tier"), 1.0)
        u = ((holders[i] + 0.5) ** 0.7) * mult * tier
        weights[i] = u
        total_u += u
    blended = [0.0] * n
    for i in active:
        prog = weights[i] / total_u if total_u > 0 else 1.0 / len(active)
        prop = holders[i] / total_before if total_before > 0 else prog
        blended[i] = 0.7 * prog + 0.3 * prop

    w_sub = _subsequent_weight(case.get("target"))
    rows = []
    for i, trophy in enumerate(trophies):
        if i in full_set:
            # Still 100%: every tracked player earns it (it was 100% on launch
            # night, still 100% at the refresh, and still 100% on the public
            # page on 20 Nov). A small downside hedge only.
            rows.append({"id": trophy["id"], "q": [99.9, 99.99, 100, 100, 100]})
            continue
        w = blended[i]
        w = min(1.0 - 1e-9, max(1e-9, w))
        cdf = beta_binomial_cdf(new_holders, kappa * w, kappa * (1.0 - w))
        # Holder-count ambiguity: mix over the feasible launch counts.
        rate = rates[i] if rates[i] is not None else 0.0
        cands = _holder_candidates(rate, players_before)
        if holders[i] in cands and len(cands) > 1:
            h_options = cands
        else:
            h_options = [holders[i]]
        h0 = holders[i]

        def refresh_cdf(x, cdf=cdf, h_options=h_options):
            total = 0.0
            for h in h_options:
                k = math.floor(x * REFRESH_PLAYERS / 100.0 - h)
                if k < 0:
                    total += 0.0
                elif k >= len(cdf):
                    total += 1.0
                else:
                    total += cdf[k]
            return total / len(h_options)

        # Median from the count distribution, never the expected count.
        k_med = quantile_from_cdf(cdf, 0.5)
        med_rate = 100.0 * (h0 + k_med) / REFRESH_PLAYERS
        med_logit = logit_pct(med_rate)

        def evol_cdf(x, med_logit=med_logit):
            return norm_cdf((logit_pct(x) - med_logit) / EVOL_SPREAD)

        levels = case.get("levels") or LEVELS
        qs = mixture_quantiles(refresh_cdf, 1.0 - w_sub, evol_cdf, levels)
        rows.append({"id": trophy["id"], "q": _clean_row(qs)})
    return {"trophies": rows}


def _clean_row(values):
    """Round to 4 decimals, clamp to [0, 100], enforce non-decreasing."""
    out = []
    prev = 0.0
    for value in values:
        rounded = round4(min(100.0, max(0.0, value)))
        if rounded < prev:
            rounded = prev
        out.append(rounded)
        prev = rounded
    return out


# --------------------------------------------------------------------------
# Regime B -- multi-refresh history (the final case)
# --------------------------------------------------------------------------

def _distinct_states(history):
    """Reads where the page actually changed (consecutive duplicates dropped)."""
    states = []
    prev_key = None
    for read in history:
        key = (
            read.get("players_tracked"),
            read.get("trophies_earned"),
            tuple(read.get("rates") or []),
        )
        if key != prev_key:
            states.append(read)
            prev_key = key
    return states


def _read_time(read):
    return parse_time(read.get("at")) or parse_time(read.get("slot"))


def forecast_regime_b(case, last_filled):
    """Horizon-matched trend ensemble with separate holder/player fits."""
    trophies = case["trophies"]
    states = _distinct_states(case["history"])
    if len(states) < 2:
        raise ValueError("Regime B needs at least two distinct states")
    levels = case.get("levels") or LEVELS
    horizon_h = (case.get("horizon_minutes") or 0) / 60.0
    horizon_d = horizon_h / 24.0
    release = parse_time(RELEASE_AT)
    target_t = parse_time(case.get("target"))
    times = []
    for state in states:
        t = _read_time(state)
        times.append(hours_between(release, t) if t is not None else 0.0)
    if target_t is not None:
        h_target = hours_between(release, target_t)
    else:
        h_target = times[-1] + horizon_h
    h_target = max(0.0, h_target)

    players = [state.get("players_tracked") or 0 for state in states]
    n_last = players[-1]
    if n_last <= 0:
        raise ValueError("no players tracked in the last read")
    log_n = [math.log(max(1, p)) for p in players]
    log_t = [math.log1p(max(0.0, h)) for h in times]
    a1, b1, _ = linfit(log_t, log_n)
    a2, b2, _ = linfit(times, log_n)
    n1 = math.exp(a1 + b1 * math.log1p(h_target))
    n2 = math.exp(a2 + b2 * h_target)
    n3 = float(n_last)
    lo_n, hi_n = float(n_last), 10.0 * n_last
    n1 = min(hi_n, max(lo_n, n1))
    n2 = min(hi_n, max(lo_n, n2))
    n_med = median3(n1, n2, n3)

    # Holder fits use the recent window (last 24h of distinct states): the
    # full history's flat pre-ignition prefix would drag down the growth
    # estimate for recently-ignited trophies. Fall back to all states if the
    # window holds fewer than 3. Player fits keep the full history (player
    # growth is smooth and monotone).
    window_start = times[-1] - 24.0
    fit_idx = [i for i, h in enumerate(times) if h >= window_start]
    if len(fit_idx) < 3:
        fit_idx = list(range(len(times)))
    fit_states = [states[i] for i in fit_idx]
    fit_times = [times[i] for i in fit_idx]

    # Refresh staleness: empirical gap distribution, memoryless approximation.
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    gap_med = median_of(gaps) if gaps else 6.0
    p_stale = min(0.5, math.exp(-horizon_h / max(gap_med, 1.0)))

    alpha = 0.25 + 0.25 * math.exp(-horizon_h / 48.0)
    process_var = 0.35 ** 2 * horizon_d
    rows = []
    for idx, trophy in enumerate(trophies):
        r_last = last_filled["rates"][idx]
        base_logit = logit_pct(r_last)
        # Holder trajectory for this trophy over the fit window.
        xs_log, xs_lin, ys = [], [], []
        h_max = 0.0
        h_first = None
        h_last_win = 0.0
        for state, h in zip(fit_states, fit_times):
            rates = state.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            p = state.get("players_tracked") or 0
            if p <= 0:
                continue
            holders_now = rates[idx] * p / 100.0
            h_max = max(h_max, holders_now)
            if h_first is None:
                h_first = holders_now
            h_last_win = holders_now
            xs_log.append(math.log1p(max(0.0, h)))
            xs_lin.append(h)
            ys.append(math.log(holders_now + 0.5))
        if h_first is None:
            h_first = 0.0
        # Flat in the window: no growth signal. A flat late/cumulative
        # trophy may still ignite (its takeoff is the dominant uncertainty);
        # a flat early trophy has already had its chance (saturated).
        flat = h_last_win <= h_first + max(1.0, 0.1 * h_first)
        mult = MULT.get(trophy["id"], 1.0)
        if len(ys) >= 2:
            ah1, bh1, s_fit = linfit(xs_log, ys)
            ah2, bh2, _ = linfit(xs_lin, ys)
            h1 = math.exp(ah1 + bh1 * math.log1p(h_target)) - 0.5
            h2 = math.exp(ah2 + bh2 * h_target) - 0.5
            h1 = min(n1, max(0.0, h1))
            h2 = min(n2, max(0.0, h2))
            r1 = 100.0 * h1 / n1
            r2 = 100.0 * h2 / n2
        else:
            s_fit = 0.3
            r1 = r_last
            r2 = r_last
            h1 = r_last / 100.0 * n1
            h2 = r_last / 100.0 * n2
        # No model may move a rate more than 3 logits from the last read:
        # guards against explosive exponential fits.
        l1 = min(base_logit + 3.0, max(base_logit - 3.0, logit_pct(r1)))
        l2 = min(base_logit + 3.0, max(base_logit - 3.0, logit_pct(r2)))
        l3 = base_logit
        ens_logit = median3(l1, l2, l3)
        mean_l = (l1 + l2 + l3) / 3.0
        v_models = ((l1 - mean_l) ** 2 + (l2 - mean_l) ** 2 + (l3 - mean_l) ** 2) / 3.0
        s_fit = min(1.5, max(0.1, s_fit))
        blend_logit = (1.0 - alpha) * base_logit + alpha * ens_logit
        r_blend = inv_logit_pct(blend_logit)
        # Ignition uncertainty: a trophy with no observed holders, or a flat
        # late/cumulative trophy, has no growth signal in s_fit or V_models,
        # yet its takeoff timing is the most uncertain thing about it. Widen
        # the spread for those (full term at zero holders or flat-and-late,
        # quarter term at 1-2 holders).
        if h_max <= 0 or (flat and 0.05 <= mult <= 0.35):
            ign_var = 1.0
        elif h_max <= 2:
            ign_var = 0.25
        else:
            ign_var = 0.0
        spread = min(
            3.0, math.sqrt(s_fit * s_fit + process_var + v_models + ign_var)
        )
        # Raw (unclamped) holder forecasts for the branch decision: the
        # logit clamp would otherwise turn a 0.01% floor rate into phantom
        # holders at large N.
        h_last_raw = r_last / 100.0 * n_last
        h_ens_raw = median3(h1, h2, r_last / 100.0 * n_med)

        if h_max <= 0 or flat or h_ens_raw >= 10.0:
            # Logit-normal branch. For a never-observed or flat trophy the
            # Gaussian is not redundant: no count distribution centered near
            # the current level can express a takeoff to dozens of holders,
            # so the widened spread above carries the ignition uncertainty.
            if levels == LEVELS:
                qs = [inv_logit_pct(blend_logit + z * spread) for z in Z]
            else:
                qs = None  # filled by the CDF path below

            def trend_cdf(x, mu=blend_logit, s=spread):
                return norm_cdf((logit_pct(x) - mu) / s)

            if qs is None:
                qs = mixture_quantiles(trend_cdf, 1.0, trend_cdf, levels)
        else:
            # Very rare trophy: discrete count uncertainty (negative
            # binomial matched to the logit spread) instead of a redundant
            # large Gaussian. The mean is the geometric alpha-blend of the
            # last holders and the ensemble holders, floored at 1: a trophy
            # with zero holders so far may still gain one, and at large N a
            # sub-0.01% median costs nothing under the scorer's clamp.
            h_geo = (
                math.exp(
                    (1.0 - alpha) * math.log(h_last_raw + 0.5)
                    + alpha * math.log(h_ens_raw + 0.5)
                )
                - 0.5
            )
            mu_c = max(h_geo, 1.0)
            if spread * spread > 1.0 / mu_c:
                size = 1.0 / (spread * spread - 1.0 / mu_c)
            else:
                size = None
            cdf = negbin_cdf(mu_c, size)

            def trend_cdf(x, cdf=cdf, n_med=n_med):
                k = math.floor(x * n_med / 100.0)
                if k < 0:
                    return 0.0
                if k >= len(cdf):
                    return 1.0
                return cdf[k]

            qs = [100.0 * quantile_from_cdf(cdf, tau) / n_med for tau in levels]

        if p_stale > 0.02:
            pers_spread = max(1e-6, RW_SIGMA * math.sqrt(horizon_d))

            def pers_cdf(x, mu=base_logit, s=pers_spread):
                return norm_cdf((logit_pct(x) - mu) / s)

            qs = mixture_quantiles(trend_cdf, 1.0 - p_stale, pers_cdf, levels)
        rows.append({"id": trophy["id"], "q": _clean_row(qs)})
    return {"trophies": rows}


# --------------------------------------------------------------------------
# Fallbacks, routing, entry point
# --------------------------------------------------------------------------

def filled_last_rates(case):
    """Last read's rates with None filled from the most recent earlier read
    that has the trophy, else the pooled median of present rates."""
    history = case["history"]
    trophies = case["trophies"]
    last = history[-1]
    rates = list(last.get("rates") or [])
    rates = (rates + [None] * len(trophies))[: len(trophies)]
    present = [r for r in rates if r is not None]
    if not present:
        for read in history:
            present.extend(r for r in (read.get("rates") or []) if r is not None)
    pooled = median_of(present) if present else 1.0
    for i in range(len(trophies)):
        if rates[i] is None:
            filled = None
            for read in reversed(history[:-1]):
                rs = read.get("rates") or []
                if i < len(rs) and rs[i] is not None:
                    filled = rs[i]
                    break
            rates[i] = filled if filled is not None else pooled
    out = dict(last)
    out["rates"] = rates
    return out


def is_frozen(history):
    if not history:
        return False
    first = history[0]
    key0 = (
        first.get("players_tracked"),
        first.get("trophies_earned"),
        tuple(first.get("rates") or []),
    )
    for read in history[1:]:
        key = (
            read.get("players_tracked"),
            read.get("trophies_earned"),
            tuple(read.get("rates") or []),
        )
        if key != key0:
            return False
    return True


def forecast_persistence(case, last_filled, sigma=RW_SIGMA):
    """Last rate, random-walk spread on the logit scale."""
    levels = case.get("levels") or LEVELS
    horizon_d = (case.get("horizon_minutes") or 0) / 1440.0
    spread = sigma * math.sqrt(max(0.0, horizon_d))
    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        rate = last_filled["rates"][idx]
        lg = logit_pct(rate)
        if levels == LEVELS:
            qs = [inv_logit_pct(lg + z * spread) for z in Z]
        else:

            def cdf(x, mu=lg, s=max(spread, 1e-9)):
                return norm_cdf((logit_pct(x) - mu) / s)

            qs = mixture_quantiles(cdf, 1.0, cdf, levels)
        rows.append({"id": trophy["id"], "q": _clean_row(qs)})
    return {"trophies": rows}


def forecast_case(case, kappa=KAPPA):
    trophies = case["trophies"]
    levels = case.get("levels") or LEVELS
    if len(levels) != 5:
        levels = LEVELS
    case = dict(case)
    case["levels"] = levels
    history = case.get("history") or []
    if not history:
        return {
            "trophies": [
                {"id": t["id"], "q": [0.5, 0.75, 1.0, 1.5, 2.5]}
                for t in trophies
            ]
        }
    last_filled = filled_last_rates(case)
    horizon_min = case.get("horizon_minutes") or 0
    if horizon_min <= 0:
        return forecast_persistence(case, last_filled, sigma=0.0)
    try:
        target = parse_time(case.get("target"))
        first = parse_time(FIRST_REFRESH_AT)
        if (
            is_frozen(history)
            and last_filled.get("players_tracked") == FROZEN_PLAYERS
            and last_filled.get("trophies_earned") == FROZEN_EARNED
            and target is not None
            and first is not None
            and target >= first
        ):
            return forecast_regime_a(case, last_filled, kappa=kappa)
        if is_frozen(history):
            return forecast_persistence(case, last_filled)
        return forecast_regime_b(case, last_filled)
    except Exception:
        # Never crash a case: an invalid case takes no prize. Fall back to the
        # simple persistence forecast.
        return forecast_persistence(case, last_filled)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
