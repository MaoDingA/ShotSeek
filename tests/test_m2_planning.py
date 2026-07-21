import json
import shutil
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from shotseek.planning.cache import PlannerCache, cache_key
from shotseek.planning.router import PlannerRouter, query_requires_model
from shotseek.planning.rules import RulePlanner, build_rule_spec
from shotseek.planning.schema import QuerySpecV2
from shotseek.planning.stepfun import StepFunPlanner, normalize_planner_response
from shotseek.retrieval.candidates import normalized_tokens

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "stepfun"
    / "query_planner_response.contract.sample.json"
)


def test_query_spec_v2_rejects_unknown_fields_and_empty_plan() -> None:
    with pytest.raises(ValidationError):
        QuerySpecV2.model_validate(
            {"raw_query": "test", "scene_id": "scene_0001"}
        )
    with pytest.raises(ValidationError):
        QuerySpecV2(raw_query="test")


def test_rule_planner_preserves_exact_dialogue() -> None:
    spec = build_rule_spec('"We have to follow our passions"')
    assert spec.quoted_text == "we have to follow our passions"
    assert spec.evidence_preference[0] == "dialogue"


@pytest.mark.parametrize(
    ("query", "entities"),
    [
        ("老爷爷", ["older", "adult", "man"]),
        ("大爷", ["older", "adult", "man"]),
        ("老奶奶", ["older", "adult", "woman"]),
    ],
)
def test_rule_planner_normalizes_colloquial_age_terms(
    query: str, entities: list[str]
) -> None:
    spec = build_rule_spec(query)
    assert [item.text for item in spec.entities] == entities
    assert spec.keywords == []


@pytest.mark.parametrize(
    ("query", "relation"),
    [
        ("woman after robot appears", "after"),
        ("woman before robot appears", "before"),
        ("woman during the bridge scene", "during"),
        ("woman between robot and rifle", "between"),
    ],
)
def test_rule_planner_supports_temporal_relations(
    query: str, relation: str
) -> None:
    spec = build_rule_spec(query)
    assert spec.temporal_constraints[0].relation == relation


@pytest.mark.parametrize(
    ("query", "value"),
    [
        ("first woman outdoors", 1),
        ("second woman outdoors", 2),
        ("最后一次女人在室外", "last"),
        ("女人第3次在室外", 3),
    ],
)
def test_rule_planner_supports_ordinals(query: str, value: int | str) -> None:
    spec = build_rule_spec(query)
    assert spec.ordinal is not None
    assert spec.ordinal.value == value


def test_router_uses_rule_for_simple_query() -> None:
    result = PlannerRouter().plan("mechanical ocular implant")
    assert result.trace.status == "RULE"
    assert result.trace.planner == "rule"
    assert result.trace.trace_id.startswith("trace_")


@pytest.mark.parametrize(
    "query",
    [
        "找到拿着收音机的男人",
        "有人按下播放按钮",
        "戴眼镜的人看着全息屏幕",
        "桥上面对机器人的年轻男人",
        "瞄准步枪的人",
    ],
)
def test_router_requires_stepfun_for_every_chinese_query(query: str) -> None:
    assert query_requires_model(query) is True


def test_router_keeps_simple_english_query_offline() -> None:
    assert query_requires_model("man holding a radio") is False


def test_chinese_query_fixture_normalizes_to_english_constraints() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = PlannerRouter().plan(
        "找到机器人出现之后女人第二次举起机械手的场景",
        fixture_response=raw,
    )
    spec = result.query_spec
    assert result.trace.planner == "stepfun"
    assert spec.raw_query.startswith("找到机器人")
    assert [item.text for item in spec.entities] == ["woman"]
    assert spec.actions == ["raise"]
    assert spec.objects == ["robotic hand"]


def test_router_uses_offline_stepfun_fixture_for_complex_query() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = PlannerRouter().plan(
        "找到机器人出现之后女人第二次举起机械手的场景",
        mode="stepfun",
        fixture_response=raw,
    )
    assert result.trace.status == "CACHED"
    assert result.trace.planner == "stepfun"
    assert result.query_spec.ordinal is not None
    assert result.query_spec.ordinal.value == 2
    assert result.query_spec.raw_query.startswith("找到机器人")


def test_router_falls_back_without_network_or_fixture() -> None:
    result = PlannerRouter().plan(
        "woman before robot appears",
        mode="stepfun",
        allow_network=False,
    )
    assert result.trace.status == "FALLBACK"
    assert result.trace.planner == "rule"
    assert result.trace.fallback_reason == "RuntimeError"


def test_content_addressed_cache_is_used_without_network() -> None:
    directory = ROOT / "runs" / "tests" / "m2-planner-cache"
    shutil.rmtree(directory, ignore_errors=True)
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    router = PlannerRouter(cache_dir=directory)
    first = router.plan(
        "woman second time after robot",
        mode="stepfun",
        fixture_response=raw,
    )
    second = router.plan("woman second time after robot", mode="cache")
    assert first.trace.planner == "stepfun"
    assert second.trace.status == "CACHED"
    assert second.trace.planner == "cache"
    assert first.query_spec == second.query_spec


def test_planner_normalizes_dict_items_to_strings() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "content": {
                        "entities": [{"text": "young man"}],
                        "actions": [{"value": "face"}],
                        "objects": [{"text": "robot arm"}],
                        "locations": [{"name": "bridge"}],
                    }
                }
            }
        ]
    }

    spec = normalize_planner_response(
        raw,
        query="机械手在年轻男人和女人之间",
        top_k=3,
    )

    assert spec.actions == ["face"]
    assert spec.objects == ["robot arm"]
    assert spec.locations == ["bridge"]
    assert "{'text':" not in spec.model_dump_json()


def test_live_planner_repairs_invalid_schema_once() -> None:
    responses = [
        {
            "choices": [
                {"message": {"content": {"entities": [], "actions": []}}}
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": {
                            "entities": [
                                {"text": "man", "role": "subject"}
                            ],
                            "objects": ["holographic display"],
                            "ordinal": {
                                "value": 1,
                                "scope": "after_temporal_filter",
                            },
                        }
                    }
                }
            ]
        },
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=responses[len(requests) - 1])

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        timeout=1.0,
    ) as client:
        result = StepFunPlanner(client=client).plan_live(
            "第一个戴眼镜的男人看全息屏幕",
            api_key="test-key",
            retry_attempts=1,
        )

    assert len(requests) == 2
    assert result.query_spec.ordinal is not None
    assert result.query_spec.ordinal.value == 1
    assert "repaired invalid model output once" in result.trace.route_reason


def test_auto_planner_uses_fast_path_for_fully_normalized_chinese() -> None:
    result = PlannerRouter().plan("爷爷")

    assert result.trace.status == "RULE"
    assert [item.text for item in result.query_spec.entities] == [
        "older",
        "adult",
        "man",
    ]


def test_rule_fallback_preserves_chinese_first_ordinal() -> None:
    spec = build_rule_spec("第一个戴眼镜的男人看全息屏幕")

    assert spec.ordinal is not None
    assert spec.ordinal.value == 1
    assert "glasses" in spec.objects
    assert "display" in spec.objects


@pytest.mark.parametrize(
    ("query", "entities"),
    [
        ("机械手在两个人中间", ["people"]),
        ("机械手在年轻男人和女人之间", ["young", "man", "woman"]),
    ],
)
def test_auto_planner_preserves_spatial_relationships(
    query: str, entities: list[str]
) -> None:
    result = PlannerRouter().plan(query)
    spec = result.query_spec

    assert result.trace.status == "RULE"
    assert [item.text for item in spec.entities] == entities
    assert normalized_tokens(" ".join(spec.objects)) == ["robot", "limb"]
    assert normalized_tokens(" ".join(spec.keywords)) == ["between"]


def test_auto_fast_path_ignores_stale_model_cache() -> None:
    query = "机械手在两个人中间"
    directory = ROOT / "runs" / "tests" / "m2-stale-planner-cache"
    shutil.rmtree(directory, ignore_errors=True)
    router = PlannerRouter(cache_dir=directory)
    PlannerCache(directory).put(
        cache_key(query, top_k=3, model=router.stepfun.model),
        QuerySpecV2(
            raw_query=query,
            entities=[{"text": "person"}],
            objects=["robot arm"],
        ),
    )

    result = router.plan(query)

    assert result.trace.status == "RULE"
    assert [item.text for item in result.query_spec.entities] == ["people"]
    assert normalized_tokens(
        " ".join(result.query_spec.keywords)
    ) == ["between"]


def test_auto_planner_preserves_behind_relationship() -> None:
    result = PlannerRouter().plan(
        "穿军装的女人站在戴机械眼男人后面"
    )
    spec = result.query_spec

    assert result.trace.status == "RULE"
    assert [item.text for item in spec.entities] == ["woman", "man"]
    assert spec.actions == ["stand"]
    assert spec.locations == ["behind"]


def test_planner_prunes_unknown_model_fields_and_invalid_relations() -> None:
    raw = {
        "choices": [{
            "message": {"content": {
                "entities": [{"text": "woman"}],
                "spatial_relations": [{"relation": "behind"}],
                "temporal_constraints": [{"relation": "near"}],
                "ordinal": {"value": 1, "scope": "invalid"},
                "unexpected": "ignored",
            }}
        }]
    }

    spec = normalize_planner_response(raw, query="女人在后面", top_k=3)

    assert [item.text for item in spec.entities] == ["woman"]
    assert spec.temporal_constraints == []
    assert spec.ordinal is not None
    assert spec.ordinal.value == 1
    assert spec.ordinal.scope == "matching_event"


def test_all_m1_queries_have_valid_rule_query_spec_v2() -> None:
    for line in (ROOT / "eval" / "m1_queries.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        query = json.loads(line)["text"]
        result = RulePlanner().plan(query)
        assert result.query_spec.schema_version == "query-v2"
