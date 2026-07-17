"""Deterministic QuerySpec v2 parser and M1 compatibility adapter."""

from __future__ import annotations

import re
from time import perf_counter

from shotseek.planning.schema import (
    AnchorSpec,
    EntityConstraint,
    NegativeConstraint,
    OrdinalConstraint,
    PlannerResult,
    PlannerTrace,
    QuerySpecV2,
    TemporalConstraint,
)
from shotseek.retrieval.query_rules import ALIASES as M1_ALIASES

PROMPT_VERSION = "rule-planner-v2"
QUOTED_RE = re.compile(r'["“](.+?)["”]')
TOKEN_RE = re.compile(r"[a-z0-9']+|[\u4e00-\u9fff]+", re.IGNORECASE)
STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "of", "to", "with", "and", "or",
    "behind", "scene", "shot", "find", "show", "me", "where", "that",
    "after", "before", "during", "between", "first", "second", "last", "not",
    "的", "在", "正", "里", "里的", "那个", "找到", "镜头", "场景",
}
ALIASES = {
    **M1_ALIASES,
    "之前": " before ",
    "当中": " during ",
    "期间": " during ",
    "第二次": " second ",
    "最后一次": " last ",
    "最后": " last ",
    "举起": " raise ",
    "走出": " exit ",
    "争吵": " argue ",
    "出现": " appear ",
    "机器人": " robot ",
    "拿枪": " holding rifle ",
    "旁边": " beside ",
    "说完": " after saying ",
    "军装女性": " woman military uniform ",
    "机械四足": " mechanical quadruped ",
    "带镜步枪": " scoped rifle ",
    "年轻男子": " young man ",
    "监控室": " control ",
    "金发": " blonde ",
    "运河": " canal ",
    "军装": " military uniform ",
    "女性": " woman ",
    "四足": " quadruped ",
    "户外": " outdoors ",
    "有人": " person ",
    "瞄着": " aiming ",
    "棕发": " brown hair ",
    "盯着": " looking ",
    "男子": " man ",
    "看向": " looking ",
    "右侧": " right ",
    "多台": " multiple ",
    "显示器": " monitor ",
    "gun operator": " person ",
    "four-legged": " quadruped ",
    "blond": " blonde ",
    "female": " woman ",
    "outside": " outdoors ",
    "soldier": " military ",
    "inside": " indoors ",
    "machine": " robot ",
    "sighting": " aiming ",
}
ENTITY_TERMS = {
    "man", "woman", "person", "young", "tom", "people", "male", "female",
    "character", "blonde", "brunette",
}
ACTION_TERMS = {
    "look", "looking", "speak", "speaking", "aim", "aiming", "reach", "reaching",
    "stand", "standing", "operate", "operating", "face", "facing", "move", "moving",
    "raise", "exit", "argue", "appear", "hold", "holding", "observe", "observes",
}
LOCATION_TERMS = {
    "indoor", "indoors", "outdoor", "outdoors", "bridge", "canal", "rooftop",
    "rooftops", "lab", "workspace", "street", "office", "room", "city", "park",
}


def expand_aliases(value: str) -> str:
    result = value.lower()
    for source in sorted(ALIASES, key=len, reverse=True):
        target = ALIASES[source]
        result = result.replace(source, f" {target} ")
    return " ".join(result.split())


def tokens(value: str, *, expand: bool = True) -> list[str]:
    result: list[str] = []
    normalized = expand_aliases(value) if expand else value
    for token in TOKEN_RE.findall(normalized):
        cleaned = token.lower().strip("'")
        if cleaned and cleaned not in STOP_WORDS and cleaned not in result:
            result.append(cleaned)
    return result


def _anchor(value: str) -> AnchorSpec:
    quoted = QUOTED_RE.search(value)
    raw_tokens = tokens(QUOTED_RE.sub(" ", value), expand=False)
    entities = [
        EntityConstraint(text=token, role="other")
        for token in raw_tokens
        if token in ENTITY_TERMS
    ]
    actions = [token for token in raw_tokens if token in ACTION_TERMS]
    locations = [token for token in raw_tokens if token in LOCATION_TERMS]
    keywords = [
        token
        for token in raw_tokens
        if token not in ENTITY_TERMS | ACTION_TERMS | LOCATION_TERMS
    ]
    return AnchorSpec(
        quoted_text=quoted.group(1).strip() if quoted else None,
        entities=entities,
        actions=actions,
        locations=locations,
        keywords=keywords,
    )


def _ordinal(expanded: str) -> OrdinalConstraint | None:
    if re.search(r"\blast\b", expanded):
        return OrdinalConstraint(value="last")
    match = re.search(r"\b(?:the\s+)?(\d+)(?:st|nd|rd|th)?\b", expanded)
    chinese = re.search(r"第\s*(\d+)\s*次", expanded)
    if chinese:
        return OrdinalConstraint(value=int(chinese.group(1)))
    if match:
        return OrdinalConstraint(value=int(match.group(1)))
    if re.search(r"\bsecond\b", expanded):
        return OrdinalConstraint(value=2)
    if re.search(r"\bfirst\b", expanded):
        return OrdinalConstraint(value=1)
    return None


def _split_temporal(expanded: str) -> tuple[str, list[TemporalConstraint]]:
    for relation in ("between", "after", "before", "during"):
        marker = f" {relation} "
        if marker not in f" {expanded} ":
            continue
        target, remainder = expanded.split(marker, 1)
        if relation == "between":
            parts = re.split(r"\s+(?:and|与|和)\s+", remainder, maxsplit=1)
            if len(parts) == 2:
                return target, [
                    TemporalConstraint(
                        relation="between",
                        anchor=_anchor(parts[0]),
                        second_anchor=_anchor(parts[1]),
                    )
                ]
        return target, [TemporalConstraint(relation=relation, anchor=_anchor(remainder))]
    return expanded, []


def _negative_constraints(expanded: str) -> tuple[str, list[NegativeConstraint]]:
    patterns = [
        r"(?:\bnot\b|不是|不要)(.+?)(?=\b(?:after|before|during|between)\b|$)",
        r"(?:exclude|排除)(.+)$",
    ]
    result: list[NegativeConstraint] = []
    cleaned = expanded
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        for token in tokens(match.group(1), expand=False):
            field = (
                "entity" if token in ENTITY_TERMS
                else "action" if token in ACTION_TERMS
                else "location" if token in LOCATION_TERMS
                else "keyword"
            )
            result.append(NegativeConstraint(field=field, text=token))
        cleaned = cleaned[: match.start()] + " " + cleaned[match.end() :]
    return cleaned, result


def build_rule_spec(query: str, *, top_k: int = 3) -> QuerySpecV2:
    raw = query.strip()
    if not raw:
        raise ValueError("query cannot be blank")
    expanded = expand_aliases(raw)
    quoted_match = QUOTED_RE.search(expanded)
    quoted_text = quoted_match.group(1).strip() if quoted_match else None
    without_quote = QUOTED_RE.sub(" ", expanded)
    without_negative, negatives = _negative_constraints(without_quote)
    target, temporal = _split_temporal(without_negative)
    target_tokens = tokens(target, expand=False)
    entities = [
        EntityConstraint(text=token, role="other")
        for token in target_tokens
        if token in ENTITY_TERMS
    ]
    actions = [token for token in target_tokens if token in ACTION_TERMS]
    locations = [token for token in target_tokens if token in LOCATION_TERMS]
    objects = [
        token
        for token in target_tokens
        if token in {"robot", "robotic", "hand", "rifle", "implant", "building", "monitor"}
    ]
    consumed = ENTITY_TERMS | ACTION_TERMS | LOCATION_TERMS | {
        "robot", "robotic", "hand", "rifle", "implant", "building", "monitor",
    }
    keywords = [token for token in target_tokens if token not in consumed]
    evidence = ["dialogue", "visual"] if quoted_text else ["visual", "dialogue"]
    return QuerySpecV2(
        raw_query=raw,
        quoted_text=quoted_text,
        entities=entities,
        actions=actions,
        objects=objects,
        locations=locations,
        keywords=keywords,
        temporal_constraints=temporal,
        ordinal=_ordinal(expanded),
        negative_constraints=negatives,
        evidence_preference=evidence,
        top_k=top_k,
    )


class RulePlanner:
    def plan(self, query: str, *, top_k: int = 3, status: str = "RULE", route_reason: str = "deterministic query") -> PlannerResult:
        started = perf_counter()
        spec = build_rule_spec(query, top_k=top_k)
        return PlannerResult(
            query_spec=spec,
            trace=PlannerTrace(
                trace_id="pending",
                status=status,
                planner="rule",
                route_reason=route_reason,
                cache_hit=False,
                fallback_reason=None,
                latency_ms=(perf_counter() - started) * 1000,
                prompt_version=PROMPT_VERSION,
            ),
        )
