"""HandFly benchmark runner: naming, reasoning tracking, cost, report."""

from llm_plays.bench import campaign_name, effort, flight_cost, load_suite, report


def test_reasoning_effort_is_part_of_every_flight_identity():
    s = load_suite("handfly-v1")
    assert effort(s, "gpt-6-sol") == "low"                               # suite default
    s["models"]["gpt-6-sol"]["reasoning"] = "high"                       # per-model override
    assert effort(s, "gpt-6-sol") == "high"
    a = campaign_name("handfly-v1", "land_sbbh13", "gpt-6-sol", "low")
    b = campaign_name("handfly-v1", "land_sbbh13", "gpt-6-sol", "high")
    assert a != b and a.endswith("@low") and b.endswith("@high")


def test_cost_uses_cached_price_and_report_shows_reasoning():
    r = {"input_tokens": 1_000_000, "cached_tokens": 800_000, "output_tokens": 10_000}
    assert abs(flight_cost(r, (2.0, 0.1, 10.0)) - (0.4 + 0.08 + 0.1)) < 1e-9
    assert flight_cost({**r, "cost_usd": 1.23}, (2.0, 0.1, 10.0)) == 1.23      # endpoint-reported cost wins
    text = report("handfly-v1", load_suite("handfly-v1"))
    assert "| Reasoning |" in text and "Reasoning tok/turn" in text and "| gpt-6-luna | low |" in text


def test_connection_failures_are_not_scored_as_flying():
    from llm_plays.bench import infra_error
    assert infra_error({"status": "error", "notes": ["model error: Connection error."]})
    assert infra_error({"status": "error", "notes": ["HTTPConnectionPool(...): Read timed out."]})
    assert not infra_error({"status": "failed", "notes": ["continued a dangerous approach below 200 ft"]})
    assert not infra_error({"status": "turn_limit", "notes": []})


def test_same_model_at_another_effort_is_its_own_entry():
    from llm_plays.bench import api_model
    s = load_suite("handfly-v1")
    assert api_model("glm-5.3-flash@high") == "glm-5.3-flash"
    assert effort(s, "glm-5.3-flash@high") == "high" and effort(s, "glm-5.3-flash") == "low"
    assert campaign_name("handfly-v1", "land_sbbh13", "glm-5.3-flash@high", "high") == \
        "bench-handfly-v1-land_sbbh13-glm-5.3-flash@high"


def test_out_of_credits_is_a_billing_error_not_a_score():
    from llm_plays.bench import billing_error, infra_error
    r = {"status": "error", "notes": ["model error: Error code: 402 - ... This request requires more credits ..."]}
    assert infra_error(r) and billing_error(r)
    assert not billing_error({"status": "error", "notes": ["model error: Connection error."]})


def test_false_landing_claims_are_flagged():
    from llm_plays.bench import false_done
    assert false_done({"status": "done", "result": "INCOMPLETE"})      # glm: "landed" at 300 ft AGL
    assert not false_done({"status": "done", "result": "PASS"})
    assert not false_done({"status": "give_up", "result": "INCOMPLETE"})
