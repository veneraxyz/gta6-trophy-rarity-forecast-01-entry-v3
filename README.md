[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/veneraxyz/gta6-trophy-rarity-forecast-01-entry-v3/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/veneraxyz/gta6-trophy-rarity-forecast-01-entry-v3/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

# gta6-trophy-rarity-forecast-01 — entry v3

Forecasts each GTA VI PS5 trophy's Exophase earn rate at the case's target as
five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95). Python 3.13, standard library
only, no network, deterministic.

## The model

**Regime A — frozen launch-night page, first refresh known (practice cases).**
The page sat at 638 players / 943 trophies earned, then refreshed to 3,868
players / 4,276 earned. The 100% trophy keeps every player, so the other 54
trophies went from 305 holders to 408: exactly +103 new holders to allocate.
v2 allocated them uniformly and scored 0.163 — a trophy gaining zero holders
should be multiplied by 638/3868 = 0.1649, not 0.2206. v3 allocates the +103
by progression (launch holders × a per-trophy accessibility multiplier from
name/description plus the public page's current rates), blended 70/30 with
proportional growth, under a Dirichlet-multinomial (κ = 100), so each
trophy's new-holder count is a beta-binomial marginal and quantiles come from
the counts. A mixture with a subsequent-evolution branch (90/10 at a 04:30
target, 70/30 at 05:00) covers the possibility that the answer read shows a
later state than the first refresh.

**Regime B — multi-refresh history (final case).** Holders and players fitted
separately with a three-model ensemble per trophy (log-time trend,
linear-time trend, persistence), blended with persistence on the logit scale
with weight α(h) = 0.25 + 0.25·exp(−h/48). Spread
s(h) = √(s_fit² + 0.35²·(h/24) + V_models), no ceiling; very rare trophies
get negative-binomial count quantiles. Refresh timing from the empirical gap
distribution.

**Fallbacks.** Frozen page with no known refresh: persistence + random-walk
spread. Missing rates: last non-null read, then pooled median. Empty history:
flat rows.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes `<id>.out` per case.
