[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zebhyrus/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zebhyrus/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at a case's target, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## The model

A trophy's rate is earners divided by players tracked, and the two counts
move for different reasons. The tracked pool explodes after launch as the
audience piles in, while a trophy's earners grow only as players reach
that trophy. So the program forecasts the two counts apart and divides
them at the target, instead of treating the rate as a curve of its own.

- Both counts are fitted in log time since release, `x = ln(1 + hours)`.
  Exophase refreshes on its own schedule, so the history repeats the
  same page for hours. It is collapsed to one row per distinct page
  state, anchored at the first read that showed it: the state was
  produced before that read, and anchoring at the last read that
  repeated it would understate how long the counts had to grow. That
  anchoring alone was worth most of the gain over the first version.
- Players tracked get a log time slope from the recent states (the last
  day's worth, or all of them when there are fewer), blended with the
  slope of the last two states. With a single state there is no slope
  to fit, so a prior slope of 1.3 stands in.
- Each trophy's earners get their own log time slope. When the history
  has enough states the slope is fitted per trophy on the recent
  states. Otherwise it comes from a prior that reads the trophy's last
  rate: trophies the early cohort earned more are the early, easy ones,
  and their earners keep growing fastest. The prior is
  `0.3 + 0.56 * (ln rate - mean ln rate)`, clipped to [0, 2].
- Both extrapolations are damped over the horizon, dividing the slope
  by `(1 + horizon_days / 4)`, and tracked growth is capped at 60x the
  last count, so multi day forecasts bend down instead of running
  away. A trophy near 100% stays near 100%: its earners are the pool
  itself.
- The spread is honest about how far ahead the forecast looks:
  `sigma = sqrt(0.3^2 + (0.35 * sqrt(horizon_days))^2)` on the logit
  scale, about 0.4 at half a day and about 0.94 at six and a half days.
  It was set from real validation errors, not from a simulation.

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

Real out of sample validation, the decisive check for this version: fit
only on the launch night reads (638 players tracked, checkpoint 03:00
and 03:30 UTC on 19 November), predict the live Exophase page about 13
hours later (crawled 16:09 UTC, most rates down 5 to 20x on dilution),
score the mean pinball loss on the logit scale.

| forecaster | loss | 90% coverage | typical miss |
|---|---|---|---|
| this version (v2) | 0.239 | 0.89 | 0.064 pts |
| previous version (v1) | 0.819 | 0.33 | 0.181 pts |
| house baseline (starter) | 2.357 | 0.05 | 0.757 pts |

Two more checks: against an interpolated 04:30 refresh as the practice
answer, v2 scores 0.28 to 0.30. On simulated final like cases (two days
of reads with refresh steps, 6.5 day horizon, rates rising toward
plateaus), v2 scores about 0.19 against the baseline's 0.28, with
medians that stay sensible (Chapter 1 near 60%, platinum near 0.25%).

What v2 still gets wrong: the fastest growing early trophies (Chapter 1,
the fishing challenge, the arcade) are under predicted by 3 to 5x, since
one snapshot cannot tell them apart from the flat majority. A wider
spread for fast growers was tried and rejected: it helped the 13 hour
check a little but wrecked the simulated final, where fitted slopes are
high for many trophies.

## License

MIT, see `LICENSE`.

<sub>Entered on [make.fun](https://make.fun/?ref=9srzvfjv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
