from __future__ import annotations

from dataclasses import replace

from guided_story_agent.v2 import (
    CanonicalSerializer,
    FingerprintBuilder,
    canonicalize_movie_plan,
    movie_plan_lineage_token,
)
from test_v2_contracts import make_plan


def test_fingerprint_is_stable_and_ignores_identity_runtime_state() -> None:
    plan = make_plan()
    fingerprint = FingerprintBuilder().build(plan)
    changed_runtime = replace(plan, plan_id="another-id", revision=99, confirmed=False)

    assert FingerprintBuilder().build(changed_runtime).value == fingerprint.value
    assert canonicalize_movie_plan(plan) == canonicalize_movie_plan(changed_runtime)
    assert fingerprint.algorithm == "sha256"


def test_fingerprint_changes_for_creative_content() -> None:
    plan = make_plan()
    changed = replace(plan, visual_style="different visual language")

    assert FingerprintBuilder().build(changed).value != FingerprintBuilder().build(plan).value


def test_canonical_serializer_is_sorted_and_lineage_token_is_deterministic() -> None:
    serializer = CanonicalSerializer()
    assert serializer.serialize({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    token_a = movie_plan_lineage_token("plan-1", 2, "f" * 64)
    token_b = movie_plan_lineage_token("plan-1", 2, "f" * 64)
    assert token_a == token_b
    assert token_a != movie_plan_lineage_token("plan-1", 3, "f" * 64)
