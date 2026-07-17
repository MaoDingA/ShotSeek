"""Deterministic query parsing for M1; model planning is intentionally deferred."""

from __future__ import annotations

import re

from shotseek.retrieval.schema import QuerySpec

QUOTED_RE = re.compile(r'["“](.+?)["”]')
TOKEN_RE = re.compile(r"[a-z0-9']+|[\u4e00-\u9fff]+", re.IGNORECASE)
STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "of", "to", "with", "and", "or",
    "behind", "scene", "shot", "find", "show", "me", "where",
    "的",
}
ALIASES = {
    "机械义眼": "mechanical ocular implant",
    "机械手": "robotic hand",
    "瞄准步枪": "scoped rifle",
    "军装女人": "woman military uniform",
    "年轻男人": "young man",
    "女人": "woman",
    "男人": "man",
    "室内": "indoor",
    "室外": "outdoor",
    "第一次": "first",
    "之后": "after",
}


def _expand_aliases(value: str) -> str:
    result = value.lower()
    for source, target in ALIASES.items():
        result = result.replace(source, f" {target} ")
    return " ".join(result.split())


def _terms(value: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(_expand_aliases(value)):
        token = token.lower().strip("'")
        if not token or token in STOP_WORDS or token in {"first", "after"}:
            continue
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def plan_query(query: str) -> QuerySpec:
    raw = query.strip()
    if not raw:
        raise ValueError("query cannot be blank")
    expanded = _expand_aliases(raw)
    quoted_match = QUOTED_RE.search(expanded)
    quoted_text = quoted_match.group(1).strip() if quoted_match else None
    residual = QUOTED_RE.sub(" ", expanded)
    temporal_relation = None
    anchor_terms: list[str] = []
    after_match = re.match(r"^(.*?)\s+after\s+(.+)$", residual, re.IGNORECASE)
    if after_match:
        residual = after_match.group(1)
        anchor_terms = _terms(after_match.group(2))
        temporal_relation = "after"
    ordinal = 1 if re.search(r"\bfirst\b", expanded, re.IGNORECASE) else None
    terms = _terms(residual)
    normalized = " ".join(expanded.split())
    return QuerySpec(
        raw_query=raw,
        normalized_query=normalized,
        quoted_text=quoted_text,
        terms=terms,
        temporal_relation=temporal_relation,
        anchor_terms=anchor_terms,
        ordinal=ordinal,
    )
