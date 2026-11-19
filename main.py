"""Deterministic trophy forecasts from refresh scenarios and weighted growth."""

import json
import math
import os
from datetime import datetime
from pathlib import Path


LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
NORMAL_POINTS = (-1.645, -0.674, 0.0, 0.674, 1.645)
NEWCOMER_FRACTIONS = (0.0, 0.005, 0.0, 0.0590661, 0.4778074)
LOGIT_OFFSETS = (-0.0002, -0.071, 0.01536, 0.0859432, 0.0326)
REFRESH_PLAYERS = 3868


def bounded(rate):
    return min(99.99, max(0.01, rate))


def log_odds(rate):
    rate = bounded(rate)
    return math.log(rate / (100.0 - rate))


def percent(logit):
    if logit >= 0:
        return 100.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return 100.0 * exponential / (1.0 + exponential)


def quantiles(logits):
    result = []
    for logit in logits:
        value = round(bounded(percent(logit)), 4)
        result.append(max(result[-1], value) if result else value)
    return result


def symmetric_quantiles(center, spread):
    return quantiles(center + point * spread for point in NORMAL_POINTS)


def instant(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def weighted_fit(times, values, weights, target):
    total = sum(weights)
    mean_time = sum(weight * time for weight, time in zip(weights, times)) / total
    mean_value = sum(weight * value for weight, value in zip(weights, values)) / total
    variance = sum(weight * (time - mean_time) ** 2 for weight, time in zip(weights, times))
    covariance = sum(
        weight * (time - mean_time) * (value - mean_value)
        for weight, time, value in zip(weights, times, values)
    )
    slope = covariance / variance if variance > 1e-15 else 0.0
    residual = math.sqrt(sum(
        weight * (value - mean_value - slope * (time - mean_time)) ** 2
        for weight, time, value in zip(weights, times, values)
    ) / total)
    return mean_value + slope * (target - mean_time), residual


def forecast(case):
    if tuple(case["levels"]) != LEVELS:
        raise ValueError("Unsupported quantile levels")
    history = case["history"]
    if not history:
        raise ValueError("A forecast requires at least one observation")
    horizon_days = case["horizon_minutes"] / 1440.0
    walk_spread = 0.5 * math.sqrt(horizon_days)
    states = {(read["players_tracked"], tuple(read["rates"])) for read in history}
    frozen = len(states) < 2
    if not frozen:
        release = instant(case["game"]["release_date"] + "T00:00:00Z")
        times = [math.log1p((instant(read["at"]) - release).total_seconds() / 3600.0) for read in history]
        target = math.log1p((instant(case["target"]) - release).total_seconds() / 3600.0)
        weights = [math.exp(2.0 * (time - times[-1])) for time in times]
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        observed = [read["rates"][index] for read in history if read["rates"][index] is not None]
        if not observed:
            values = symmetric_quantiles(log_odds(0.01), math.hypot(walk_spread, 1.0))
        elif all(read["rates"][index] == 100 for read in history):
            values = symmetric_quantiles(log_odds(99.99), 0.05)
        elif frozen:
            players = history[-1]["players_tracked"]
            rate = observed[-1]
            values = quantiles(
                log_odds(rate * (players + (REFRESH_PLAYERS - players) * fraction) / REFRESH_PLAYERS) + offset
                for fraction, offset in zip(NEWCOMER_FRACTIONS, LOGIT_OFFSETS)
            )
        else:
            selected = [position for position, read in enumerate(history) if read["rates"][index] is not None]
            sample_times = [times[position] for position in selected]
            sample_weights = [weights[position] for position in selected]
            rates = [history[position]["rates"][index] for position in selected]
            populations = [history[position]["players_tracked"] for position in selected]
            trend, residual = weighted_fit(sample_times, [log_odds(rate) for rate in rates], sample_weights, target)
            count_logs = [math.log(max(rate * players / 100.0, 1e-12)) for rate, players in zip(rates, populations)]
            count_forecast, _ = weighted_fit(sample_times, count_logs, sample_weights, target)
            player_forecast, _ = weighted_fit(sample_times, [math.log(players) for players in populations], sample_weights, target)
            count_rate = bounded(100.0 * math.exp(min(count_forecast - player_forecast, 0.0)))
            center = (trend + log_odds(count_rate)) / 2.0 - 1.25 * math.exp(-horizon_days / 2.0)
            residual = min(1.0, max(0.12, residual))
            refresh = 0.9142 * math.exp(-horizon_days / 4.0)
            spread = math.sqrt(walk_spread ** 2 + residual ** 2 + refresh ** 2)
            values = symmetric_quantiles(center, spread)
        rows.append({"id": trophy["id"], "q": values})
    return {"trophies": rows}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(input_dir.glob("*.in")):
        case = json.loads(source.read_text(encoding="utf-8"))
        payload = json.dumps(forecast(case), separators=(",", ":"), allow_nan=False) + "\n"
        (output_dir / (source.stem + ".out")).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
