[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zebhyrus/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zebhyrus/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at a case's target, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## The model

A trophy's rate is earners divided by players tracked, and early on the
tracked pool grows much faster than any trophy's earners: launch night
had 638 players tracked, the first refresh 3,868, and by launch
afternoon the live page showed most rates down 5-20x. So the program
models the two counts separately and divides them, instead of treating
the rate as a smooth curve of its own.

- Each count is fitted in log time since release, `x = ln(1 + hours
  since 2026-11-19T00:00Z)`: log players-tracked is linear in `x`, log
  earners per trophy is linear in `x`, shrunk toward its tier's mean
  slope, then extrapolated with a damped step in `x` so multi-day
  growth bends down instead of running away.
- Exophase refreshes on its own schedule, so reads repeat the same page
  for hours: the history is collapsed to one row per distinct page
  state, anchored at the last read that showed it.
- When the history has never shown a refresh (one distinct snapshot),
  the program expects the next one to be a step jump in the tracked
  pool, calibrated to launch night's first refresh (638 to 3,868), an
  expectation that fades over days as growth turns smooth. Earners ride
  that jump in proportion to how common the trophy already is.
- The spread is honest about refresh shocks: fit residuals, plus a
  random-walk term in the horizon, plus a shock term that is large when
  no refresh has been seen and fades as the horizon lengthens.

Python 3.13 standard library only, no network, deterministic: two runs
on the same case write the same bytes.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` file in the input directory and writes one
`<id>.out` per case to the output directory:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`,
one row per trophy, five finite numbers from 0 to 100, non-decreasing.

## Results so far

On the practice cases (which straddle Exophase's first refresh), scored
against the refresh under a range of earner-growth assumptions, this
forecaster scores about 0.23-0.31 pinball loss against the house
baseline's 1.01-1.69. On a simulated final-like case (multi-day horizon,
real growth curves, refresh steps) it scores about 0.35 against the
baseline's 0.51.

## License

MIT, see `LICENSE`.

<sub>Entered on [make.fun](https://make.fun/?ref=9srzvfjv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
