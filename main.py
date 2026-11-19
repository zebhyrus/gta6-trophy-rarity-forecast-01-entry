"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Model: earners and tracked players are modelled separately, then divided.

A trophy's Exophase rate is earners / players-tracked.  Early on the
tracked pool grows much faster than any trophy's earners (launch night:
638 tracked; first refresh: 3,868; by the afternoon of launch day the
live page showed most rates down 5-20x), so a rate is not a smooth curve
of its own: it is the ratio of two growing counts.  Each count is fitted
in log time since release, x = ln(1 + hours since 2026-11-19T00:00Z):

  * log players-tracked is linear in x (a power law in 1 + t);
  * log earners per trophy is linear in x, shrunk toward its tier's
    mean slope, then extrapolated with a damped step in x so growth
    bends down over a multi-day horizon instead of running away.

Quantiles are honest about the refresh shocks: the spread combines the
fit residuals, a random-walk term in the horizon, and a shock term that
is large when the history has never shown a refresh (one distinct
snapshot) and fades as the horizon lengthens.

Standard library only.  Deterministic: no randomness, no clock, no
network; the same case always writes the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Normal quantiles for the scored levels; any other level is computed.
Z_TABLE = {0.05: -1.6448536269514722, 0.25: -0.6744897501960817,
           0.5: 0.0, 0.75: 0.6744897501960817, 0.95: 1.6448536269514722}

# Fallback slope of log tracked-players against log time when the
# history shows no growth at all (one distinct snapshot).  The first
# refresh took tracked players 638 -> 3,868 in about 4.5 hours, and the
# live page by launch afternoon implied tens of thousands: a slope near
# 1.5 in log time fits that early climb.
PRIOR_TRACKED_SLOPE = 1.5
# Earners grow slower than the tracked pool on average (dilution), and
# a trophy with no readable history is assumed to grow at this share
# of the tracked slope.
PRIOR_EARNER_SHARE = 0.7


def parse_time(text):
    """ISO 8601 timestamp -> aware datetime (accepts a trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(when):
    """Hours from release to `when`, floored at 0 (pre-release reads)."""
    return max(0.0, (when - RELEASE).total_seconds() / 3600.0)


def log_time(when):
    """Log-time coordinate: ln(1 + hours since release)."""
    return math.log1p(hours_since_release(when))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 1.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    if p in Z_TABLE:
        return Z_TABLE[p]
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def ols(xs, ys):
    """Least-squares line y = a + b*x; returns (a, b, residual_std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2
                              for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid


def distinct_snapshots(case):
    """History collapsed to one row per distinct page state.

    Exophase refreshes on its own schedule, so reads repeat the same
    page for hours; only a change in players-tracked or any rate is a
    new observation.  Each row is (log-time, players-tracked, rates).
    """
    rows = []
    seen = None
    n_tr = len(case.get("trophies", []))
    for read in case.get("history", []):
        rates = list(read.get("rates") or ())
        if len(rates) < n_tr:
            rates += [None] * (n_tr - len(rates))
        key = (read.get("players_tracked"), tuple(rates))
        when = log_time(parse_time(read["at"]))
        if key == seen:
            # Same page again: the state is now known to persist until
            # this read, so anchor it at the latest time it was seen.
            rows[-1] = (when, rows[-1][1], rows[-1][2])
            continue
        seen = key
        rows.append((when, read["players_tracked"], rates))
    return rows


def damped_step(dx):
    """A step in log time, bent down for far targets: growth saturates."""
    return dx / (1.0 + 0.35 * dx) if dx > 0 else 0.0


def fit_tracked(rows, x_target):
    """Forecast players-tracked at x_target; returns (n_target, slope).

    Anchored at the last read and extrapolated with the same damped
    step the earner fits use, so a trophy whose earners track the pool
    keeps its rate instead of diluting away.
    """
    last_n = rows[-1][1]
    x_last = rows[-1][0]
    step = damped_step(max(0.0, x_target - x_last))
    if len(rows) < 2:
        slope = PRIOR_TRACKED_SLOPE
        log_n = math.log(max(1.0, last_n)) + slope * step
        return min(max(math.exp(log_n), last_n), last_n * 500.0), slope
    xs = [r[0] for r in rows]
    ys = [math.log(max(1.0, r[1])) for r in rows]
    _, b, _ = ols(xs, ys)
    if b <= 0.0:
        b = PRIOR_TRACKED_SLOPE
    log_n = math.log(max(1.0, last_n)) + b * step
    n_target = math.exp(log_n)
    return min(max(n_target, last_n), last_n * 500.0), b


def fit_earners(rows, trophies, x_target, tracked_slope):
    """Per-trophy earner forecasts.

    Returns a list, one per trophy, of (earners_target, sigma_fit):
    earners fitted as log-linear in log time, shrunk toward the tier
    mean slope, extrapolated with a damped step so multi-day growth
    bends down, and floored at the last observed earners (earners
    never decrease).  sigma_fit is the fit's residual scale.
    """
    n_tr = len(trophies)
    last = rows[-1]
    last_n = max(1.0, last[1])
    # Earners per trophy at each distinct snapshot.
    earners = []
    for r in rows:
        n = max(1.0, r[1])
        earners.append([max(0.5, (rate if rate is not None else 0.0) * n / 100.0)
                        for rate in r[2]])
    xs = [r[0] for r in rows]
    x_last = xs[-1]
    dx = max(0.0, x_target - x_last)
    # Damped extrapolation step: growth in log time saturates, so a
    # far target moves less per unit of log time than a near one.
    # The tracked-players fit uses the same step, so ratios stay put
    # for trophies whose earners track the pool.
    dx_eff = damped_step(dx)

    raw = []  # (slope, intercept, resid, last_earners) per trophy
    for i in range(n_tr):
        ys = [math.log(e[i]) for e in earners]
        if len(rows) >= 3:
            a, b, resid = ols(xs, ys)
        else:
            a, b, resid = ys[-1] - tracked_slope * PRIOR_EARNER_SHARE * x_last, \
                tracked_slope * PRIOR_EARNER_SHARE, 0.0
        raw.append((a, b, resid, earners[-1][i]))

    # Tier mean slopes from the trophies that had a real fit.
    tier_slopes = {}
    for tier in {t.get("tier") for t in trophies}:
        fitted = [raw[i][1] for i, t in enumerate(trophies)
                  if t.get("tier") == tier and len(rows) >= 3]
        if fitted:
            tier_slopes[tier] = sum(fitted) / len(fitted)
    global_slope = tracked_slope * PRIOR_EARNER_SHARE
    if len(rows) >= 3:
        fitted_all = [raw[i][1] for i in range(n_tr)]
        global_slope = sum(fitted_all) / len(fitted_all)

    out = []
    for i, trophy in enumerate(trophies):
        a, b, resid, e_last = raw[i]
        tier_mean = tier_slopes.get(trophy.get("tier"), global_slope)
        # A trophy nearly everyone has earned (rate near 100%) can only
        # grow with the tracked pool itself: its earners track N, so its
        # slope floor rises with its last rate toward the tracked slope.
        last_rate = last[2][i] if last[2][i] is not None else 0.0
        share_floor = 0.7 + 0.3 * (min(100.0, last_rate) / 100.0) ** 2
        slope_floor = tracked_slope * share_floor
        if len(rows) >= 3:
            w = len(rows) / (len(rows) + 4.0)
            slope = w * b + (1.0 - w) * tier_mean
            slope = max(slope, slope_floor)
            # Re-anchor at the last observation so shrinkage moves the
            # slope, never the level the history actually ends at.
            log_e = math.log(e_last) + slope * dx_eff
            sig = min(max(resid, 0.15), 1.2)
        else:
            slope = max(tier_mean if tier_mean is not None else global_slope,
                        slope_floor)
            log_e = math.log(e_last) + slope * dx_eff
            sig = 0.6
        e_target = max(math.exp(log_e), e_last)
        out.append((e_target, sig))
    return out


def forecast_case(case):
    """One case -> {"trophies": [{"id": n, "q": [...]}, ...]}."""
    trophies = case["trophies"]
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    target = parse_time(case["target"])
    x_target = log_time(target)
    horizon_days = max(0.0, case.get("horizon_minutes", 0) / 1440.0)
    rows = distinct_snapshots(case)
    if not rows:
        return {"trophies": [{"id": t["id"], "q": [0.05, 0.2, 0.5, 1.2, 3.0]}
                             for t in trophies]}

    n_target, tracked_slope = fit_tracked(rows, x_target)
    jump = 1.0
    if len(rows) <= 1:
        # No refresh has ever been seen: the next one is a step jump in
        # the tracked pool, not smooth growth (launch night's first
        # refresh took it 638 -> 3,868, about 6x including the smooth
        # climb).  Expect such a jump soon after release; the
        # expectation fades over days, when growth turns smooth.
        jump = 1.0 + 3.0 * math.exp(-horizon_days / 1.0)
        n_target = min(n_target * jump, rows[-1][1] * 50.0)
    earner_rows = fit_earners(rows, trophies, x_target, tracked_slope)

    # Refresh-shock term: when the history has never shown a refresh,
    # the next one can move every rate at once (the practice cases are
    # exactly this: one page, then a refresh).  It bites on short
    # horizons and fades over days, when refreshes average out.
    if len(rows) <= 1:
        shock = 0.85 * math.exp(-horizon_days / 2.0)
    elif len(rows) <= 3:
        shock = 0.4 * math.exp(-horizon_days / 2.0)
    else:
        shock = 0.15 * math.exp(-horizon_days / 2.0)
    walk = 0.30 * math.sqrt(horizon_days)

    out_rows = []
    last_rates = rows[-1][2]
    for i, trophy in enumerate(trophies):
        e_target, sig_fit = earner_rows[i]
        if jump > 1.0:
            # A refresh records earners too: a trophy nearly everyone
            # has rides the jump with the pool, a rare one barely
            # moves, so its earners take the jump to the power of the
            # square root of its last rate.
            last_rate = last_rates[i] if last_rates[i] is not None else 0.0
            e_target *= jump ** math.sqrt(min(100.0, last_rate) / 100.0)
        # Earners cannot exceed the tracked pool.
        e_target = min(e_target, 0.999 * n_target)
        median = clamp_rate(100.0 * e_target / n_target)
        sigma = math.sqrt(sig_fit ** 2 + walk ** 2 + shock ** 2)
        sigma = min(max(sigma, 0.25), 2.0)
        centre = logit(median)
        qs = []
        prev = None
        for level in levels:
            q = clamp_rate(100.0 * sigmoid(centre + norm_ppf(level) * sigma))
            q = round4(q)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        out_rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": out_rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
