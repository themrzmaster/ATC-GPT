#!/usr/bin/env python3
"""
HandFly benchmark runner: every model x every scenario of a suite (llm_plays/benchmark.yaml), one flight each,
sequential (there is one simulator). Finished flights are skipped, so an interrupted run just continues.

    python -m llm_plays.bench --suite handfly-v1 --dry-run
    python -m llm_plays.bench --suite handfly-v1 [--models a,b] [--scenarios x,y]
    python -m llm_plays.bench --suite handfly-v1 --report
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGNS = ROOT / "campaigns"
SUITES = Path(__file__).with_name("benchmark.yaml")

# rough per-flight token use, from the flights so far (history of 24 turns, one screenshot kept)
EST_TOKENS = {"in": 650_000, "out": 5_000, "visual_in": 950_000}


def load_suite(name: str) -> dict:
    suites = yaml.safe_load(SUITES.read_text())
    if name not in suites:
        raise SystemExit(f"unknown suite {name}; have {list(suites)}")
    s = suites[name]
    s["models"] = {m: c or {} for m, c in s["models"].items() if (c or {}).get("enabled", True)}
    return s


def api_model(key: str) -> str:
    """Suite entries can be `model@effort` to fly the same model at another reasoning effort."""
    return key.split("@", 1)[0]


def effort(s: dict, key: str) -> str:
    """Requested reasoning effort: `reasoning:` in the entry, else the `@effort` in its key, else the suite setting."""
    cfg = s["models"][key]
    return str(cfg.get("reasoning") or (key.split("@", 1)[1] if "@" in key else None)
               or (s.get("settings") or {}).get("reasoning", "low"))


def campaign_name(suite: str, scenario: str, model: str, reasoning: str) -> str:
    # the effort is part of the name: the same model at another effort is a different entry
    return f"bench-{suite}-{scenario}-{api_model(model)}@{reasoning}"


def flight_cost(r: dict, price) -> float:
    if r.get("cost_usd"):
        return float(r["cost_usd"])
    p_in, p_cache, p_out = price
    cached = r.get("cached_tokens") or 0
    return ((r.get("input_tokens", 0) - cached) * p_in + cached * p_cache + r.get("output_tokens", 0) * p_out) / 1e6


def result(suite: str, scenario: str, model: str, reasoning: str):
    p = CAMPAIGNS / campaign_name(suite, scenario, model, reasoning) / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def estimate(s: dict, todo: list) -> float:
    total = 0.0
    for sc, m in todo:
        p_in, _, p_out = s["models"][m]["price"]
        tin = EST_TOKENS["visual_in"] if "visual" in sc else EST_TOKENS["in"]
        total += (tin * p_in + EST_TOKENS["out"] * p_out) / 1e6
    return total


def false_done(r: dict) -> bool:
    """The model reported the goal achieved but the examiner disagrees (e.g. 'landed' while still at 300 ft)."""
    return r.get("status") == "done" and r.get("result") not in ("PASS", "LANDED")


def report(name: str, s: dict) -> str:
    models, scen = list(s["models"]), s["scenarios"]
    short = {sc: sc.replace("land_", "") for sc in scen}
    lines = [f"## {name}: {s.get('description', '')}", "",
             "| Model | Reasoning | " + " | ".join(short[sc] for sc in scen) +
             " | **Mean** | Pass | False claims | Reasoning tok/turn | Out tok/turn | Cache | Cost |",
             "|---|---|" + "---|" * len(scen) + "---|---|---|---|---|---|---|"]
    # Mean = over the flights flown so far (Pass shows n/flown): a partial row is labelled, not padded with zeros
    rows = []
    for m in models:
        eff = effort(s, m)
        cells, scores, passes, cost, rtok, turns, claims = [], [], 0, 0.0, 0, 0, 0
        out_tok, all_turns, tin, tcached = 0, 0, 0, 0
        for sc in scen:
            r = result(name, sc, m, eff)
            if not r:
                cells.append("·")
                continue
            scores.append(r["score"])
            passes += r["result"] == "PASS"
            cost += flight_cost(r, s["models"][m]["price"])
            if r.get("reasoning_tokens") is not None:
                rtok += r["reasoning_tokens"]
                turns += r.get("turns", 0)
            tag = {"PASS": "", "LANDED": " L", "INCOMPLETE": " inc", "FAIL": " FAIL", "CRASH": " CRASH"}[r["result"]]
            claims += false_done(r)
            out_tok += r.get("output_tokens", 0)
            all_turns += r.get("turns", 0)
            tin += r.get("input_tokens", 0)
            tcached += r.get("cached_tokens") or 0
            cells.append(f"{r['score']:.0f}{tag}" + ("!" if false_done(r) else ""))
        mean = sum(scores) / len(scores) if scores else 0.0
        if not scores:
            continue                                               # not flown yet: no row
        rt = f"{rtok / turns:.0f}" if turns else "n/a"          # 0 = the effort was not applied (or not reported)
        rows.append((mean, f"| {api_model(m)} | {eff} | " + " | ".join(cells) +
                     f" | **{mean:.1f}**{'' if len(scores) == len(scen) else f' ({len(scores)}/{len(scen)} flown)'} | {passes}/{len(scores)} | {claims} | {rt}"
                     f" | {out_tok / max(all_turns, 1):.0f} | {100 * tcached / max(tin, 1):.0f}% | ${cost:.2f} |"))
    lines += [r for _, r in sorted(rows, key=lambda x: -x[0])]
    lines += ["", "Reasoning = requested effort; tok/turn = reasoning tokens the model actually used per turn "
              "(n/a: flown before this was tracked; 0: the endpoint ignored it or doesn't report it - Claude via the "
              "proxy thinks but reports 0, so Out tok/turn, which includes thinking, is the better signal). "
              "Cache = share of input tokens served from the prompt cache.",
              "Score 0-100 per flight: 70 for the milestone ladder + 30 for the outcome (landing quality, or the "
              "missed approach when below minimums). FAIL/CRASH = 0. inc = incomplete, L = landed but not every gate. "
              "! = false claim: the model reported the goal achieved when it wasn't (e.g. 'landed' while airborne)."]
    return "\n".join(lines)


def ensure_sim(wait_s: int = 90) -> bool:
    """The SSH tunnel to the box drops now and then (seen: a Mac network blip mid-run): reopen it."""
    import time
    from llm_plays.sim import XPlaneSim
    if XPlaneSim().health():
        return True
    print("sim not reachable: reopening the tunnel", flush=True)
    subprocess.run(["pkill", "-f", "ssh.*-L 5066"], check=False)
    subprocess.Popen([str(ROOT / "scripts" / "box.sh"), "tunnel"], cwd=ROOT,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        time.sleep(5)
        if XPlaneSim().health():
            return True
    return False


BILLING = ("Error code: 402", "requires more credits", "insufficient_quota", "exceeded your current quota")


def infra_error(r: dict) -> bool:
    """The flight ended on a connection or billing problem (not the model's flying): it doesn't count."""
    return r.get("status") == "error" and any(
        "Connection" in n or "timed out" in n or any(b in n for b in BILLING) for n in r.get("notes") or [])


def billing_error(r: dict) -> bool:
    """Out of credits at the provider: retrying won't help, stop and tell the human."""
    return r.get("status") == "error" and any(b in n for n in r.get("notes") or [] for b in BILLING)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="handfly-v1")
    ap.add_argument("--models", help="comma list (default: all enabled in the suite)")
    ap.add_argument("--scenarios", help="comma list (default: all in the suite)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    s = load_suite(args.suite)
    if args.report:
        print(report(args.suite, s))
        return
    models = args.models.split(",") if args.models else list(s["models"])
    models.sort(key=lambda m: s["models"][m]["price"][0])      # cheapest first: problems surface cheaply
    scen = args.scenarios.split(",") if args.scenarios else s["scenarios"]
    todo = [(sc, m) for m in models for sc in scen if not result(args.suite, sc, m, effort(s, m))]
    print(f"{args.suite}: {len(todo)} flights to fly, est. ${estimate(s, todo):.2f}")
    for sc, m in todo:
        print(f"  {api_model(m) + '@' + effort(s, m):26s} {sc}")
    if args.dry_run:
        return
    st = s.get("settings") or {}
    for i, (sc, m) in enumerate(todo, 1):
        cfg = s["models"][m]
        cmd = [sys.executable, "-m", "llm_plays.run", "--mission", sc, "--model", api_model(m), "--fresh",
               "--campaign", campaign_name(args.suite, sc, m, effort(s, m)), "--reasoning", effort(s, m),
               "--keep-turns", str(st.get("keep_turns", 24)), "--spectate-every", str(st.get("spectate_every", 2))]
        if cfg.get("base_url"):
            cmd += ["--base-url", os.path.expandvars(cfg["base_url"])]   # e.g. ${LLM_PROXY_URL}
        if cfg.get("history"):
            cmd += ["--history", cfg["history"]]
        if cfg.get("cache_marks"):
            cmd += ["--cache-marks"]
        if cfg.get("images"):
            cmd += ["--images", cfg["images"]]
        print(f"\n[{i}/{len(todo)}] {m} / {sc}", flush=True)
        rc, r, sim_down = 0, None, False
        for attempt in range(2):
            if not ensure_sim():
                sim_down = True
                break
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            r = result(args.suite, sc, m, effort(s, m))
            if r and billing_error(r):
                break
            if not (r and infra_error(r) and attempt == 0):
                break
            d = CAMPAIGNS / campaign_name(args.suite, sc, m, effort(s, m))
            (CAMPAIGNS / "_invalid").mkdir(exist_ok=True)
            d.rename(CAMPAIGNS / "_invalid" / (d.name + "-infra-error"))
            r = None
            print(f"[{i}/{len(todo)}] connection error mid-flight: archived, flying it again", flush=True)
        if r and infra_error(r):                  # still a connection/billing error: archive it, don't score it
            d = CAMPAIGNS / campaign_name(args.suite, sc, m, effort(s, m))
            (CAMPAIGNS / "_invalid").mkdir(exist_ok=True)
            d.rename(CAMPAIGNS / "_invalid" / (d.name + ("-billing" if billing_error(r) else "-infra-error")))
            print(f"stopping: {'the provider is out of credits' if billing_error(r) else 'connection errors'} "
                  f"({m}) - fix and re-run to continue")
            break
        if sim_down:
            print("stopping: the simulator is not reachable (is X-Plane running on the box?) - re-run to continue")
            break
        print(f"[{i}/{len(todo)}] {m} / {sc}: " + (
            f"{r['result']} {r['score']} | {r.get('turns', 0)} turns | cache "
            f"{100 * (r.get('cached_tokens') or 0) / max(r.get('input_tokens', 0), 1):.0f}% | "
            f"${flight_cost(r, cfg['price']):.2f}" if r else f"no result (exit {rc})"), flush=True)
        if not r:
            print("stopping: the flight failed to run (API error?) - fix and re-run to continue")
            break
    print("\n" + report(args.suite, s))


if __name__ == "__main__":
    main()
