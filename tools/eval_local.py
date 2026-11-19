"""Offline structural, reproducibility, and synthetic-answer evaluation."""

import argparse
import copy
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KIT = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/gta6-trophy-rarity-forecast-01-kit")
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
POINTS = (-1.645, -0.674, 0.0, 0.674, 1.645)
SCENARIOS = (0.0, 0.1, 0.25, 0.5)
TARGETS = (0.062107, 0.161250, 0.402393, 0.664930)
ROUND_ONE = (0.064122, 0.219820, 0.572941, 1.010501)


def logit(rate):
    probability = min(99.99, max(0.01, rate)) / 100.0
    return math.log(probability / (1.0 - probability))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def validate(payload, case):
    assert len(payload) <= 65536, "Output exceeds byte limit"
    data = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object)
    assert type(data) is dict and set(data) == {"trophies"}, "Invalid outer object"
    rows = data["trophies"]
    assert type(rows) is list and len(rows) == len(case["trophies"]), "Wrong row count"
    for row, trophy in zip(rows, case["trophies"]):
        assert type(row) is dict and set(row) == {"id", "q"}, "Invalid row keys"
        assert type(row["id"]) is int and row["id"] == trophy["id"], "Wrong id or ordering"
        values = row["q"]
        assert type(values) is list and len(values) == 5, "Wrong quantile count"
        assert all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100 for value in values), "Invalid quantile"
        assert all(left <= right for left, right in zip(values, values[1:])), "Crossing quantiles"
    return data


def run_program(program, cases, python):
    payloads = []
    runtimes = []
    with tempfile.TemporaryDirectory(prefix="trophy-eval-") as temporary:
        directory = Path(temporary)
        inputs = directory / "input"
        inputs.mkdir()
        for name, case in cases.items():
            (inputs / (name + ".in")).write_text(json.dumps(case), encoding="utf-8")
        for repeat in range(2):
            outputs = directory / f"output-{repeat}"
            environment = dict(os.environ, MAKEFUN_INPUT_DIR=str(inputs), MAKEFUN_OUTPUT_DIR=str(outputs), PYTHONHASHSEED="0")
            started = time.perf_counter()
            completed = subprocess.run([python, "-P", "-s", str(program)], env=environment, capture_output=True, timeout=8, check=False)
            elapsed = time.perf_counter() - started
            assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
            assert elapsed < 8, "Runtime limit exceeded"
            assert {path.name for path in outputs.iterdir()} == {name + ".out" for name in cases}, "Unexpected output files"
            current = {}
            for name, case in cases.items():
                path = outputs / (name + ".out")
                assert path.is_file() and not path.is_symlink(), "Output must be a regular file"
                payload = path.read_bytes()
                validate(payload, case)
                current[name] = payload
            payloads.append(current)
            runtimes.append(elapsed * 1000.0)
    assert payloads[0] == payloads[1], "Non-deterministic output"
    return {name: validate(payload, cases[name]) for name, payload in payloads[0].items()}, runtimes, {name: len(payload) for name, payload in payloads[0].items()}


def synthetic_answers(case, fraction):
    last = case["history"][-1]
    return [
        100.0 if all(read["rates"][index] == 100 for read in case["history"])
        else rate * (last["players_tracked"] + (3868 - last["players_tracked"]) * fraction) / 3868
        for index, rate in enumerate(last["rates"])
    ]


def score(output, answers):
    losses = []
    covered = []
    for row, answer in zip(output["trophies"], answers):
        for level, estimate in zip(LEVELS, row["q"]):
            difference = logit(answer) - logit(estimate)
            losses.append(2.0 * max(level * difference, (level - 1.0) * difference))
        covered.append(row["q"][0] <= answer <= row["q"][-1])
    return statistics.mean(losses), statistics.mean(covered)


def final_case(template, saturating=False):
    case = copy.deepcopy(template)
    release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    checkpoint = release + timedelta(days=2, hours=11, minutes=30)
    case.update(id="synthetic-final", at=checkpoint.isoformat(), horizon_minutes=6510,
                target=(checkpoint + timedelta(minutes=6510)).isoformat(), history=[])
    for hour in range(0, 49, 2):
        rates = []
        for index, trophy in enumerate(case["trophies"]):
            if trophy["id"] == 2:
                rates.append(100.0)
            elif saturating:
                rates.append(0.5 + 90.0 * (hour / 48.0) ** 2)
            else:
                scale = 0.6 + index / 100.0
                rates.append(scale * (1.0 + hour) ** 0.65)
        stamp = (release + timedelta(hours=hour)).isoformat()
        case["history"].append({"slot": stamp, "at": stamp, "players_tracked": 638 + hour * 100, "trophies_earned": 0, "rates": rates})
    return case


def reference_live(case, index):
    release = datetime.fromisoformat(case["game"]["release_date"] + "T00:00:00+00:00")
    def axis(stamp):
        return math.log1p((datetime.fromisoformat(stamp.replace("Z", "+00:00")) - release).total_seconds() / 3600.0)
    times = [axis(read["at"]) for read in case["history"]]
    target = axis(case["target"])
    weights = [math.exp(2.0 * (sample - times[-1])) for sample in times]
    def fit(values):
        average_time = statistics.fmean(sample * weight for sample, weight in zip(times, weights)) / statistics.fmean(weights)
        average_value = sum(value * weight for value, weight in zip(values, weights)) / sum(weights)
        slope = sum(weight * (sample - average_time) * (value - average_value) for sample, value, weight in zip(times, values, weights)) / sum(weight * (sample - average_time) ** 2 for sample, weight in zip(times, weights))
        deviation = math.sqrt(sum(weight * (value - average_value - slope * (sample - average_time)) ** 2 for sample, value, weight in zip(times, values, weights)) / sum(weights))
        return average_value + slope * (target - average_time), deviation
    rates = [read["rates"][index] for read in case["history"]]
    populations = [read["players_tracked"] for read in case["history"]]
    trend, deviation = fit([logit(rate) for rate in rates])
    count, _ = fit([math.log(rate * players / 100.0) for rate, players in zip(rates, populations)])
    population, _ = fit([math.log(players) for players in populations])
    count_rate = min(99.99, max(0.01, 100.0 * math.exp(min(count - population, 0.0))))
    days = case["horizon_minutes"] / 1440.0
    center = (trend + logit(count_rate)) / 2.0 - 1.25 * math.exp(-days / 2.0)
    spread = math.sqrt(0.25 * days + min(1.0, max(0.12, deviation)) ** 2 + (0.9142 * math.exp(-days / 4.0)) ** 2)
    return [round(min(99.99, max(0.01, 100.0 / (1.0 + math.exp(-center - point * spread)))), 4) for point in POINTS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    parser.add_argument("--python", default=sys.executable)
    arguments = parser.parse_args()
    cases = {name: json.loads((arguments.kit / "input" / (name + ".in")).read_text()) for name in ("p00", "p01")}
    programs = {"entry": ROOT / "main.py", "starter": arguments.kit / "starter" / "main.py"}
    forecasts = {}
    print("Validation: isolated single-case runs, strict JSON, ordered ids, limits, two-run byte comparison")
    print("program case valid deterministic bytes run1_ms run2_ms")
    for label, program in programs.items():
        forecasts[label] = {}
        for name, case in cases.items():
            result, runtimes, sizes = run_program(program, {name: case}, arguments.python)
            forecasts[label][name] = result[name]
            print(f"{label:7} {name:4} PASS  PASS {sizes[name]:5} {runtimes[0]:8.3f} {runtimes[1]:8.3f}")
    print("\nSynthetic refresh scenarios (means over p00 and p01; coverage uses raw percent answers)")
    print("phi   entry_loss starter_loss entry_cov starter_cov target   delta     round1   improved")
    agreements = []
    for fraction, target, previous in zip(SCENARIOS, TARGETS, ROUND_ONE):
        metrics = {label: [score(forecasts[label][name], synthetic_answers(case, fraction)) for name, case in cases.items()] for label in programs}
        entry_loss, entry_coverage = map(statistics.mean, zip(*metrics["entry"]))
        starter_loss, starter_coverage = map(statistics.mean, zip(*metrics["starter"]))
        agreements.append(entry_loss <= previous and abs(entry_loss - target) < 0.0001)
        print(f"{fraction:4.2f} {entry_loss:10.6f} {starter_loss:12.6f} {entry_coverage:9.2%} {starter_coverage:11.2%} {target:.6f} {entry_loss-target:+.6f} {previous:.6f} {entry_loss < previous}")
    case = final_case(cases["p00"])
    output, _, _ = run_program(programs["entry"], {"final": case}, arguments.python)
    rising = [(index, row) for index, row in enumerate(output["final"]["trophies"]) if row["id"] != 2]
    assert all(row["q"][2] > case["history"][-1]["rates"][index] for index, row in rising), "Rising medians did not extrapolate"
    spreads = [(logit(row["q"][4]) - logit(row["q"][0])) / 3.29 for _, row in rising]
    mean_spread = statistics.mean(spreads)
    assert 0.8 < mean_spread < 1.3, "Final spread is not around one logit"
    for index, row in rising:
        assert row["q"] == reference_live(case, index), "Live branch differs from independent formula"
    print(f"\nSynthetic final: {len(rising)}/{len(rising)} rising medians above last rate; mean implied spread {mean_spread:.6f}; independent formula PASS")
    stress = final_case(cases["p00"], saturating=True)
    stress_output, _, _ = run_program(programs["entry"], {"saturation": stress}, arguments.python)
    assert stress_output["saturation"]["trophies"][0]["q"] == reference_live(stress, 0), "Saturating count extrapolation differs from specified clamp"
    print("Saturating count extrapolation: PASS")
    edge = copy.deepcopy(cases["p00"])
    for read in edge["history"]:
        read["rates"][0] = None
        read["rates"][2] = 0.0
    run_program(programs["entry"], {"missing": edge, "practice": cases["p01"]}, arguments.python)
    live_edge = copy.deepcopy(case)
    for read in live_edge["history"]:
        read["rates"][0] = None
        read["rates"][2] = 0.0
        read["rates"][3] = None
    live_edge["history"][-1]["rates"][3] = 0.5
    run_program(programs["entry"], {"live-edge": live_edge}, arguments.python)
    print("Multiple input files; missing, zero, saturated, and single-observation trophies: PASS")
    assert all(agreements), "Scenario results disagree with targets or fail round-one comparison; constants unchanged"
    print(f"All checks PASS using {arguments.python}. Scenario losses are hypothetical; real answers are unavailable.")


if __name__ == "__main__":
    main()
