#!/usr/bin/env python3
"""
LLM plays X-Plane - run one model on one mission ("beat the game").

    scripts/box.sh tunnel                                   # API of PI_LLMPlays on localhost:5066
    python -m llm_plays.run --mission land_sbbh13 --model gpt-5.6-luna --fresh
    python -m llm_plays.run --campaign land_sbbh13-gpt-5.6-luna     # resume where it stopped
    python -m llm_plays.viewer                              # watch it live: http://127.0.0.1:8341
    python -m llm_plays.run --leaderboard                   # compare every campaign

A campaign is campaigns/<mission>-<model>[-tag]/ plus a saved X-Plane situation of the same name.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CAMPAIGNS = ROOT / "campaigns"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="land_sbbh13")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--reasoning", default="low", help="reasoning effort (low|medium|high|none)")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible endpoint (e.g. a LiteLLM proxy at https://<host>/v1); key from $LLM_PROXY_KEY")
    ap.add_argument("--campaign", help="campaign name (default <mission>-<model>[-tag])")
    ap.add_argument("--tag", default="")
    ap.add_argument("--fresh", action="store_true", help="start the mission from scratch")
    ap.add_argument("--max-turns", type=int)
    ap.add_argument("--keep-turns", type=int, default=24, help="turns of history sent to the model (cost knob)")
    ap.add_argument("--history", choices=("sliding", "blocks"), default="sliding",
                    help="sliding = v1 (drop one turn per turn); blocks = prompt-cache friendly (see agent.Pilot)")
    ap.add_argument("--cache-marks", action="store_true", help="Anthropic prompt-cache marks (Claude models)")
    ap.add_argument("--images", choices=("latest", "keep"), default="latest",
                    help="latest = v1 (only the newest screenshot, swapped every turn); keep = cache friendly")
    ap.add_argument("--spectate-every", type=int, default=2,
                    help="viewer-only screenshot every N acts (0 = off); no tokens, ~6 s wall time each")
    ap.add_argument("--leaderboard", action="store_true")
    ap.add_argument("--make-start", metavar="SCENARIO|all",
                    help="build start state(s) (place + weather + engine start) and save them as start situations")
    ap.add_argument("--list", action="store_true", help="list scenarios (llm_plays/scenarios.yaml)")
    ap.add_argument("--save-start", metavar="MISSION",
                    help="save the sim's CURRENT state as this mission's start situation (set it up in X-Plane first)")
    args = ap.parse_args()

    if args.leaderboard:
        print(leaderboard())
        return
    if args.list:
        from llm_plays.missions import MISSIONS, weather_report
        for name, m in MISSIONS.items():
            s = m["setup"]
            where = f"{s['airport']} {s['runway']}" + (f" -> {m['destination'][0]} {m['destination'][1]}"
                                                        if m["destination"] != (s["airport"], s["runway"]) else "")
            print(f"{name:26s} {m['type']:13s} {where:18s} {weather_report(m['weather']) or 'calm'}")
        return
    if args.make_start:
        from llm_plays.missions import MISSIONS, setup_mission, start_situation_name
        from llm_plays.sim import XPlaneSim
        from llm_plays.agent import instruments
        sim = XPlaneSim()
        names = list(MISSIONS) if args.make_start == "all" else [args.make_start]
        for n in names:
            try:
                rw = setup_mission(sim, MISSIONS[n])      # placement path (no mission_name -> no .sit load)
                sim.pause()
                sim.save(start_situation_name(n))
                m = MISSIONS[n]
                txt = instruments(sim.state(), rw, m.get("approach", "rnav"), m.get("ils_tuned"))
                print(f"OK   {n}\n     " + "\n     ".join(txt.splitlines()[:12]))
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {n:26s} {e}")
        return
    if args.save_start:
        from llm_plays.missions import MISSIONS, start_situation_name
        from llm_plays.sim import XPlaneSim
        from llm_plays.agent import instruments
        from llm_plays.missions import MilestoneTracker
        if args.save_start not in MISSIONS:
            raise SystemExit(f"unknown mission; choose from {list(MISSIONS)}")
        sim = XPlaneSim()
        sim.pause()
        print(sim.save(start_situation_name(args.save_start)))
        print(instruments(sim.state(), MilestoneTracker(MISSIONS[args.save_start]).runway))
        return

    from openai import OpenAI
    from llm_plays.missions import MISSIONS, MilestoneTracker, setup_mission
    from llm_plays.sim import XPlaneSim
    from llm_plays.agent import Campaign, Pilot

    sim = XPlaneSim()
    import time as _time
    for _ in range(6):                  # a just-reopened tunnel can take a few seconds to carry traffic
        if sim.health():
            break
        _time.sleep(5)
    else:
        raise SystemExit("PI_LLMPlays API not reachable on http://127.0.0.1:5066 - run scripts/box.sh tunnel "
                         "(and make sure X-Plane is running with the plugin)")
    name = args.campaign or "-".join(x for x in (args.mission, args.model, args.tag) if x)
    camp = Campaign(CAMPAIGNS, name)
    meta = camp.read("meta.json") or {}
    mission_name = meta.get("mission", args.mission)
    model = meta.get("model", args.model)
    mission = MISSIONS[mission_name]
    tracker = MilestoneTracker(mission)

    resumed = False
    if not args.fresh and meta and sim.load(name):
        resumed = True
        tracker.restore(camp.read("progress.json") or {})
        sim.wait_ready(settle_s=5)
        # X-Plane doesn't restore a running engine from a situation: restart it holding the pilot's last inputs
        last = next((e["args"] for e in reversed(_events(camp)) if e.get("kind") == "act"), {})
        held = {k: last.get(k) for k in ("pitch", "roll", "yaw", "trim", "throttle", "mixture", "flaps")
                if last.get(k) is not None}
        st = sim.start_engine(2.5, **held)
        sim.pause()
        print(f"resumed campaign {name} at turn {meta.get('turns', 0)} (milestones {tracker.p.reached}, "
              f"engine {'running' if st.get('engine_running') else 'NOT running'}, {st.get('agl_ft', 0):.0f} ft AGL)")
    else:
        for f in camp.dir.glob("*.jsonl"):       # events + trajectory start over on a fresh run
            f.unlink()
        meta = {"mission": mission_name, "model": model, "reasoning": args.reasoning, "endpoint": args.base_url or "openai",
                "history": args.history, "cache_marks": args.cache_marks, "images": args.images,
                "created": datetime.now().isoformat(timespec="seconds"), "turns": 0}
        print(f"setting up mission {mission_name} ...")
        setup_mission(sim, mission, mission_name=mission_name)
        sim.save(name)
    camp.write("meta.json", meta)
    camp.log("start", mission=mission_name, model=model, resumed=resumed)

    if args.base_url:
        if not os.environ.get("LLM_PROXY_KEY"):
            raise SystemExit("--base-url needs the key in $LLM_PROXY_KEY (never stored in files)")
        client = OpenAI(base_url=args.base_url, api_key=os.environ["LLM_PROXY_KEY"])
    else:
        client = OpenAI()
    pilot = Pilot(client, model, sim, mission, camp, tracker,
                  reasoning=None if args.reasoning == "none" else args.reasoning, keep_turns=args.keep_turns,
                  spectate_every=args.spectate_every, history=args.history, cache_marks=args.cache_marks,
                  images=args.images)
    if resumed:
        notes = [e["message"] for e in _events(camp) if e.get("kind") == "report"][-5:]
        pilot.first = [{"role": "user", "content": [{"type": "input_text", "text":
                        "The flight was paused and is now resumed from where you left off. Your last notes: "
                        + (" | ".join(notes) or "none") + ". Start with look()."}]}]

    def on_turn(p):
        s = p.stats
        prog = p.tracker.score()
        print(f"  turn {s.turns:3d}  acts {s.acts:3d}  sim {s.sim_s:6.1f}s  model {s.model_s:6.1f}s  "
              f"milestones {prog['milestones']}  score {prog['score']}")

    if resumed:
        pilot.stats.turns = int(meta.get("turns", 0))      # keep numbering (and the turn cap) across sessions
    max_turns = args.max_turns or mission["max_turns"]
    stats = pilot.fly(max_turns=max_turns, max_sim_s=mission["max_sim_s"], on_turn=on_turn)
    result = {"campaign": name, "mission": mission_name, "model": model, "reasoning": args.reasoning,
              **tracker.score(), "status": stats.status, "turns": stats.turns, "acts": stats.acts,
              "looks": stats.looks, "views": stats.views, "sim_s": round(stats.sim_s, 1), "model_s": round(stats.model_s, 1),
              "wall_s": round(stats.wall_s, 1), "input_tokens": stats.input_tokens,
              "output_tokens": stats.output_tokens, "cached_tokens": stats.cached_tokens,
              "reasoning_tokens": stats.reasoning_tokens, "history": args.history, "images": args.images,
              "cost_usd": round(stats.cost_usd, 5) if stats.cost_usd else None,
              "endpoint": args.base_url or "openai", "notes": stats.notes[-5:],
              "finished": datetime.now().isoformat(timespec="seconds")}
    meta["turns"] = stats.turns
    camp.write("meta.json", meta)
    camp.write("result.json", result)
    camp.log("end", **result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _events(camp):
    p = camp.dir / "events.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def leaderboard() -> str:
    rows = []
    for rp in sorted(CAMPAIGNS.glob("*/result.json")):
        rows.append(json.loads(rp.read_text()))
    if not rows:
        return "no finished campaigns yet"
    rows.sort(key=lambda r: (r["mission"], -r["score"]))
    out = ["| Mission | Model | Score | Milestones | Status | Touchdown sink | Turns | Sim s | Model s | Tokens in/out |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        td = r.get("touchdown") or {}
        sink = f"{td['sink_fpm']:.0f} fpm" if td else "-"
        out.append(f"| {r['mission']} | {r['model']} ({r.get('reasoning')}) | {r['score']} | {r['milestones']} | "
                   f"{r['status']} | {sink} | {r['turns']} | {r['sim_s']} | {r['model_s']} | "
                   f"{r['input_tokens']}/{r['output_tokens']} |")
    return "\n".join(out)


if __name__ == "__main__":
    main()
