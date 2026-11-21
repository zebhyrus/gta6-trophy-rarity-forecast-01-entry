"""Forecast Grand Theft Auto VI trophy earn rates on Exophase.

A trophy's rate is two counts divided: the players who earned it over the
players Exophase tracks for the game. The two counts move for different
reasons, so this program forecasts them apart and divides at the target.

The tracked pool explodes after launch as the audience piles in, while a
trophy's earners grow only as players reach that trophy. Launch night
showed the split starkly: 638 players tracked, then a first refresh to
3,868, and by the next afternoon the live page had most rates down
5-20x while a few early trophies (Chapter 1, the fishing challenge, the
arcade) had their earners grow tenfold or more. So:

- Both counts are fitted in log time since release, x = ln(1 + hours),
  on the distinct page states in the history. A page that Exophase has
  not refreshed repeats for hours, so the history is collapsed to one
  row per distinct state, anchored at the first read that showed it:
  the state predates that read, and anchoring at the last read would
  understate how long the counts had to grow.
- Tracked players get a log-time slope from the recent states (the last
  day's worth, or all of them when there are fewer), blended with the
  slope of the last two states. With a single state there is no slope
  to fit, so a prior slope of 1.3 stands in: the launch-week audience
  grows roughly that fast in log time.
- Each trophy's earners get their own log-time slope. When the history
  has enough states the slope is fitted per trophy on the recent
  states; otherwise it comes from a prior that reads the trophy's last
  rate: trophies the launch-night cohort earned more are the early,
  easy ones, and their earners keep growing fastest. The prior is
  0.3 + 0.56 * (ln rate - mean ln rate), clipped to [0, 2].
- Both extrapolations are damped over the horizon, dividing the slope
  by (1 + horizon_days / 4): growth bends down over days instead of
  running away, and tracked growth is capped at 60x the last count.
  A trophy near 100% stays near 100%: its earners are the pool itself.
- The spread is honest about how far ahead the forecast looks:
  sigma = sqrt(0.3^2 + (0.35 * sqrt(horizon_days))^2) on the logit
  scale, about 0.4 at half a day and about 0.94 at six and a half days.

Python 3.13 standard library only, no network, deterministic: two runs
on the same case write the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Tuned on a real out-of-sample check: fit only on launch-night reads,
# predict the live Exophase page ~13 hours later (see README).
TRACKED_PRIOR_SLOPE = 1.3
DAMPING_DAYS = 4.0
MAX_TRACKED_GROWTH = 60.0
EARNER_BASE_SLOPE = 0.3
EARNER_RATE_SLOPE = 0.56
EARNER_SLOPE_MIN = 0.0
EARNER_SLOPE_MAX = 2.0
FIT_WINDOW_HOURS = 24.0
SPREAD_BASE = 0.3
SPREAD_PER_ROOT_DAY = 0.35
SPREAD_UNIVERSAL = 0.15


def _when(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _log_clock(moment):
    """Log time since release: ln(1 + hours), floored at release."""
    hours = (moment - RELEASE).total_seconds() / 3600.0
    return math.log1p(max(hours, 0.0))


def _least_squares(xs, ys):
    """Return (intercept, slope) of the least-squares line through the points."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    spread_xx = sum((x - mean_x) ** 2 for x in xs)
    if spread_xx == 0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / spread_xx
    return mean_y - slope * mean_x, slope


def _logit(pct):
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def _pct(logit_value):
    return 100.0 / (1.0 + math.exp(-logit_value))


def _round4(value):
    """Round to 4 decimals, half up, as a plain number (100 stays 100)."""
    scaled = value * 10000.0
    whole = math.floor(scaled + 0.5)
    rounded = whole / 10000.0
    return int(rounded) if rounded.is_integer() else rounded


def _states(case):
    """Collapse the history to one row per distinct page state.

    Returns a list of (players_tracked, rates tuple, first_seen), oldest
    first. A state is anchored at the first read that showed it, because
    Exophase refreshed the page before that read, not after the last
    read that repeated it.
    """
    rows = []
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if rows and rows[-1][0] == key:
            continue
        rows.append((key, _when(read["at"])))
    return [(key[0], key[1], seen) for key, seen in rows]


def _recent(rows):
    """Indexes of the states inside the fitting window before the last one."""
    last_seen = rows[-1][2]
    keep = [j for j, (_, _, seen) in enumerate(rows)
            if (last_seen - seen).total_seconds() / 3600.0 <= FIT_WINDOW_HOURS]
    if len(keep) < 2:
        return list(range(len(rows)))
    return keep


def _tracked_at_target(rows, clocks, x_target, horizon_days):
    """Forecast the players-tracked count at the target's log time."""
    players_last = max(rows[-1][0], 1)
    keep = _recent(rows)
    if len(keep) >= 2:
        xs = [clocks[j] for j in keep]
        ys = [math.log(max(rows[j][0], 1)) for j in keep]
        _, slope_all = _least_squares(xs, ys)
        _, slope_recent = _least_squares(xs[-2:], ys[-2:])
        slope = 0.5 * slope_all + 0.5 * slope_recent
    else:
        slope = TRACKED_PRIOR_SLOPE
    slope = max(slope, 0.0)
    damp = 1.0 / (1.0 + horizon_days / DAMPING_DAYS)
    grown = slope * damp * max(x_target - clocks[-1], 0.0)
    grown = min(grown, math.log(MAX_TRACKED_GROWTH))
    return players_last * math.exp(grown)


def _earner_slopes(rows, clocks, last_rates):
    """One log-time slope per trophy for how its earners grow.

    Fitted per trophy on the recent states when the history has enough
    of them; otherwise the rate prior: early, common trophies grow
    fastest, rare late ones barely move.
    """
    count = len(last_rates)
    log_rates = [math.log(max(rate, 0.01)) for rate in last_rates]
    mean_log_rate = sum(log_rates) / count
    prior = []
    for log_rate in log_rates:
        slope = EARNER_BASE_SLOPE + EARNER_RATE_SLOPE * (log_rate - mean_log_rate)
        prior.append(min(max(slope, EARNER_SLOPE_MIN), EARNER_SLOPE_MAX))
    keep = _recent(rows)
    if len(keep) < 2:
        return prior
    fitted = {}
    for i in range(count):
        points = []
        usable = True
        for j in keep:
            rate = rows[j][1][i]
            if rate is None:
                usable = False
                break
            earners = max(rate / 100.0 * max(rows[j][0], 1), 0.5)
            points.append(math.log(earners))
        if usable:
            _, slope = _least_squares([clocks[j] for j in keep], points)
            fitted[i] = min(max(slope, EARNER_SLOPE_MIN), EARNER_SLOPE_MAX)
    return [fitted.get(i, prior[i]) for i in range(count)]


def _last_known_rates(rows, count):
    """Each trophy's rate in the newest state that lists it."""
    found = [None] * count
    for _, rates, _ in reversed(rows):
        for i, rate in enumerate(rates):
            if found[i] is None and rate is not None:
                found[i] = rate
        if all(rate is not None for rate in found):
            break
    return [rate if rate is not None else 0.0 for rate in found]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    rows = _states(case)
    clocks = [_log_clock(seen) for _, _, seen in rows]
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = _log_clock(_when(case["target"]))
    players_target = _tracked_at_target(rows, clocks, x_target, horizon_days)
    players_last = max(rows[-1][0], 1)
    trophies = case["trophies"]
    last_rates = _last_known_rates(rows, len(trophies))
    slopes = _earner_slopes(rows, clocks, last_rates)
    damp = 1.0 / (1.0 + horizon_days / DAMPING_DAYS)
    step = damp * max(x_target - clocks[-1], 0.0)
    spread = math.sqrt(SPREAD_BASE ** 2
                       + (SPREAD_PER_ROOT_DAY * math.sqrt(horizon_days)) ** 2)
    out_rows = []
    for trophy, rate_last, slope in zip(trophies, last_rates, slopes):
        if rate_last >= 90.0:
            # Near-universal trophies move with the pool: the rate stays put.
            middle = min(rate_last, 99.99)
            sigma = SPREAD_UNIVERSAL
        else:
            earners_last = max(rate_last / 100.0 * players_last, 0.5)
            earners_target = earners_last * math.exp(slope * step)
            middle = min(max(100.0 * earners_target / players_target, 0.01), 99.99)
            sigma = spread
        centre = _logit(middle)
        quantiles = [_round4(min(max(_pct(centre + z * sigma), 0.0), 100.0))
                     for z in Z_SCORES]
        quantiles.sort()
        out_rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": out_rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
