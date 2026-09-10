from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from copy import deepcopy
from typing import Any, Iterable

from .logical_form import normalize_relation


VARIANTS = (
    "baseline",
    "family_structure",
    "relation_set",
    "relation_slot_branch",
    "direction",
    "all_factors",
)

VARIANT_DEFINITIONS = {
    "baseline": "original question-only retriever order",
    "family_structure": "anonymous anchored dependency graph; relation and direction masked",
    "relation_set": "unordered core gold relation identities",
    "relation_slot_branch": "core relation identities plus undirected slot/branch dependencies (SR)",
    "direction": "SR plus KG subject/object direction (SRD)",
    "all_factors": "SRD plus explicit LF answer-type and operator constraints",
}


def _factor_usable(factor: dict[str, Any] | None) -> bool:
    return bool(
        factor
        and factor.get("parse_status") in {"parsed", "partial"}
        and factor.get("oracle_eligible", True)
    )


def _triple_tuple(value: Any) -> tuple[str, str, str]:
    if len(value) < 3:
        raise ValueError(f"Malformed scored triple: {value!r}")
    return str(value[0]), str(value[1]), str(value[2])


def _candidate_rows(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    triples = sample.get("scored_triples") or sample.get("scored_triplets") or []
    for rank, value in enumerate(triples):
        head, relation, tail = _triple_tuple(value)
        try:
            score = float(value[3]) if len(value) >= 4 else float(len(triples) - rank)
        except (TypeError, ValueError):
            score = float(len(triples) - rank)
        rows.append({
            "index": rank,
            "head": head,
            "relation_raw": relation,
            "relation": normalize_relation(relation),
            "tail": tail,
            "original_score": score,
            "original_rank": rank,
        })
    return rows


def _topic_distances(rows: list[dict[str, Any]], topics: Iterable[str]) -> dict[str, int]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        adjacency[row["head"]].add(row["tail"])
        adjacency[row["tail"]].add(row["head"])
    distances: dict[str, int] = {}
    queue: deque[str] = deque()
    for topic in topics:
        if topic in adjacency and topic not in distances:
            distances[topic] = 0
            queue.append(topic)
    while queue:
        node = queue.popleft()
        for neighbour in adjacency.get(node, ()):
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    return distances


def _best_dependency_match(
    rows: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    topics: set[str],
    *,
    use_relations: bool,
    use_direction: bool,
    state_limit: int = 200_000,
) -> dict[str, Any]:
    """Find one maximal anchored partial query-graph homomorphism.

    Candidate edges are never injected or edited.  Without direction, each KG
    edge may bind to a slot in either orientation; with direction, subject and
    object dependencies must bind to the stored KG orientation.  Shared
    dependency IDs enforce intermediate-variable and conjunction consistency.
    """
    usable_slots = [
        slot for slot in slots
        if slot.get("subject_dependency") is not None
        and slot.get("object_dependency") is not None
    ]
    if not usable_slots:
        return {
            "matched_slots": set(),
            "candidate_to_slots": {},
            "branch_any": 0,
            "branch_complete": 0,
            "full": None,
            "truncated": False,
        }
    dependency_roles: dict[str, str] = {}
    for slot in usable_slots:
        dependency_roles[str(slot["subject_dependency"])] = str(
            slot.get("subject_dependency_role") or "unknown"
        )
        dependency_roles[str(slot["object_dependency"])] = str(
            slot.get("object_dependency_role") or "unknown"
        )

    # Prompt creation removes exact duplicate triples.  Do the same here so a
    # duplicated PTH row cannot satisfy two logical slots.
    unique_indices = []
    seen_triples = set()
    for index, row in enumerate(rows):
        triple = (row["head"], row["relation_raw"], row["tail"])
        if triple not in seen_triples:
            seen_triples.add(triple)
            unique_indices.append(index)

    def role_allows(role: str | None, value: str) -> bool:
        return role != "topic" or value in topics

    options: dict[str, list[tuple[int, tuple[tuple[str, str], ...]]]] = {}
    for slot in usable_slots:
        slot_id = slot["slot_id"]
        subject_dep = str(slot["subject_dependency"])
        object_dep = str(slot["object_dependency"])
        subject_role = slot.get("subject_dependency_role")
        object_role = slot.get("object_dependency_role")
        slot_options = []
        for index in unique_indices:
            row = rows[index]
            if use_relations and row["relation"] != slot.get("relation"):
                continue
            orientations = [(row["head"], row["tail"])]
            if not use_direction and row["head"] != row["tail"]:
                orientations.append((row["tail"], row["head"]))
            for subject_value, object_value in orientations:
                if not role_allows(subject_role, subject_value):
                    continue
                if not role_allows(object_role, object_value):
                    continue
                if subject_dep == object_dep and subject_value != object_value:
                    continue
                slot_options.append((index, (
                    (subject_dep, subject_value),
                    (object_dep, object_value),
                )))
        options[slot_id] = slot_options

    # Grow from anchored/scarce slots, then prefer slots sharing an already
    # introduced dependency.  This keeps the exact search small for Top-500.
    remaining = list(usable_slots)
    ordered_slots = []
    introduced: set[str] = set()
    while remaining:
        chosen = min(
            remaining,
            key=lambda slot: (
                -sum(
                    str(slot.get(key)) in introduced
                    for key in ("subject_dependency", "object_dependency")
                ),
                -sum(
                    slot.get(key) == "topic"
                    for key in ("subject_dependency_role", "object_dependency_role")
                ),
                len(options[slot["slot_id"]]),
                slot["slot_id"],
            ),
        )
        remaining.remove(chosen)
        ordered_slots.append(chosen)
        introduced.update({
            str(chosen["subject_dependency"]),
            str(chosen["object_dependency"]),
        })

    branches: dict[str, set[str]] = defaultdict(set)
    for slot in usable_slots:
        branches[str(slot.get("branch_id", slot["slot_id"]))].add(slot["slot_id"])

    best_slots: set[str] = set()
    best_map: dict[int, set[str]] = {}
    best_score: tuple[int, int, int, int] = (-1, -1, -1, -10**18)
    state_count = 0
    truncated = False

    def consider(matched: dict[str, int]) -> None:
        nonlocal best_slots, best_map, best_score
        matched_ids = set(matched)
        complete = sum(required <= matched_ids for required in branches.values())
        any_branch = sum(bool(required & matched_ids) for required in branches.values())
        rank_cost = sum(rows[index]["original_rank"] for index in matched.values())
        score = (len(matched_ids), complete, any_branch, -rank_cost)
        if score > best_score:
            best_score = score
            best_slots = matched_ids
            candidate_map: dict[int, set[str]] = defaultdict(set)
            for slot_id, candidate_index in matched.items():
                candidate_map[candidate_index].add(slot_id)
            best_map = dict(candidate_map)

    def search(
        position: int,
        bindings: dict[str, str],
        used_candidates: set[int],
        matched: dict[str, int],
    ) -> None:
        nonlocal state_count, truncated
        if truncated:
            return
        state_count += 1
        if state_count > state_limit:
            truncated = True
            return
        consider(matched)
        if position >= len(ordered_slots):
            return
        if len(matched) + len(ordered_slots) - position < len(best_slots):
            return
        slot = ordered_slots[position]
        slot_id = slot["slot_id"]
        for candidate_index, pairs in options[slot_id]:
            if candidate_index in used_candidates:
                continue
            additions: dict[str, str] = {}
            compatible = True
            for dependency, value in pairs:
                if dependency in bindings:
                    if bindings[dependency] != value:
                        compatible = False
                        break
                elif dependency in additions:
                    if additions[dependency] != value:
                        compatible = False
                        break
                else:
                    if dependency_roles.get(dependency) == "topic" and any(
                        other_dependency != dependency
                        and dependency_roles.get(other_dependency) == "topic"
                        and other_value == value
                        for other_dependency, other_value in (
                            *bindings.items(), *additions.items()
                        )
                    ):
                        compatible = False
                        break
                    additions[dependency] = value
            if not compatible:
                continue
            bindings.update(additions)
            used_candidates.add(candidate_index)
            matched[slot_id] = candidate_index
            search(position + 1, bindings, used_candidates, matched)
            matched.pop(slot_id)
            used_candidates.remove(candidate_index)
            for dependency in additions:
                bindings.pop(dependency, None)
        # Partial matching is required when a pool is missing a gold edge.
        search(position + 1, bindings, used_candidates, matched)

    search(0, {}, set(), {})
    complete_count = sum(required <= best_slots for required in branches.values())
    any_count = sum(bool(required & best_slots) for required in branches.values())
    full: float | None
    if len(best_slots) == len(usable_slots):
        full = 1.0
    elif truncated:
        full = None
    else:
        full = 0.0
    return {
        "matched_slots": best_slots,
        "candidate_to_slots": best_map,
        "branch_any": any_count,
        "branch_complete": complete_count,
        "full": full,
        "truncated": truncated,
    }


def annotate_candidates(
    sample: dict[str, Any],
    factor: dict[str, Any] | None,
    match_variant: str | None = None,
    match_state_limit: int = 100_000,
) -> list[dict[str, Any]]:
    rows = _candidate_rows(sample)
    topics = sample.get("q_entity_in_graph") or sample.get("q_entity") or []
    distances = _topic_distances(rows, (str(topic) for topic in topics))
    node_degree: Counter[str] = Counter()
    for row in rows:
        node_degree[row["head"]] += 1
        node_degree[row["tail"]] += 1
    factor = factor or {}
    all_factor_slots = factor.get("relation_slots", []) if _factor_usable(factor) else []
    # Relation-set and relation-slot ablations intentionally exclude explicit
    # type/operator constraints; those enter only the joint all_factors arm.
    slots = [slot for slot in all_factor_slots if slot.get("role", "relation") == "relation"]
    expected_depths = {
        int(slot["topic_depth"])
        for slot in slots
        if slot.get("topic_depth") is not None
    }
    expected_directions = {
        (slot.get("topic_depth"), slot.get("topic_direction"))
        for slot in slots
        if slot.get("topic_direction") not in {None, "unknown"}
    }
    gold_relations = set(factor.get("relations", [])) if slots else set()
    answer_types = {str(value) for value in factor.get("answer_types", [])}
    family = factor.get("query_family", "other")
    branch_count = int(factor.get("branch_count", 0))

    for row in rows:
        dh = distances.get(row["head"], math.inf)
        dt = distances.get(row["tail"], math.inf)
        if math.isinf(dh) and math.isinf(dt):
            depth = None
            direction = "unknown"
        elif dh < dt:
            depth = int(dh)
            direction = "forward"
        elif dt < dh:
            depth = int(dt)
            direction = "inverse"
        else:
            depth = int(dh)
            direction = "lateral"
        row["topic_depth"] = depth
        row["topic_direction"] = direction
        row["relation_match"] = row["relation"] in gold_relations
        depth_match = (
            depth is not None
            and (
                depth in expected_depths
                or (not expected_depths and depth < max(1, int(factor.get("max_hop", 0))))
            )
        )
        junction_support = max(node_degree[row["head"]], node_degree[row["tail"]]) >= max(2, branch_count)
        if family == "single":
            row["structure_match"] = depth == 0
        elif family in {"conjunction", "mixed"}:
            row["structure_match"] = depth_match and junction_support
        else:
            row["structure_match"] = depth_match
        row["junction_support"] = junction_support
        row["direction_match"] = (
            (depth, direction) in expected_directions
            or (None, direction) in expected_directions
        )
        row["slot_matches"] = [
            slot["slot_id"]
            for slot in slots
            if row["relation"] == slot.get("relation")
            and (slot.get("topic_depth") is None or depth == slot.get("topic_depth"))
        ]
        row["slot_direction_matches"] = [
            slot["slot_id"]
            for slot in slots
            if row["relation"] == slot.get("relation")
            and (slot.get("topic_depth") is None or depth == slot.get("topic_depth"))
            and slot.get("topic_direction") in {None, "unknown", direction}
        ]
        row["branch_matches"] = sorted({
            slot["branch_id"] for slot in slots if slot["slot_id"] in row["slot_matches"]
        })
        row["all_factor_slot_matches"] = [
            slot["slot_id"]
            for slot in all_factor_slots
            if row["relation"] == slot.get("relation")
            and (slot.get("topic_depth") is None or depth == slot.get("topic_depth"))
            and (
                slot.get("role") != "type"
                or bool(answer_types & {row["head"], row["tail"]})
            )
        ]
        row["all_factor_slot_direction_matches"] = [
            slot["slot_id"]
            for slot in all_factor_slots
            if slot["slot_id"] in row["all_factor_slot_matches"]
            and slot.get("topic_direction") in {None, "unknown", direction}
        ]
        row["all_factor_branch_matches"] = sorted({
            slot["branch_id"] for slot in all_factor_slots
            if slot["slot_id"] in row["all_factor_slot_matches"]
        })
        row["answer_type_match"] = bool(answer_types & {row["head"], row["tail"]})

    topic_set = {str(topic) for topic in topics}
    empty_match = {"candidate_to_slots": {}, "truncated": False}
    anonymous_match = (
        _best_dependency_match(
            rows, slots, topic_set, use_relations=False, use_direction=False,
            state_limit=match_state_limit,
        )
        if match_variant == "family_structure" else empty_match
    )
    slot_graph_match = (
        _best_dependency_match(
            rows, slots, topic_set, use_relations=True, use_direction=False,
            state_limit=match_state_limit,
        )
        if match_variant == "relation_slot_branch" else empty_match
    )
    directed_graph_match = (
        _best_dependency_match(
            rows, slots, topic_set, use_relations=True, use_direction=True,
            state_limit=match_state_limit,
        )
        if match_variant in {"direction", "all_factors"} else empty_match
    )
    for index, row in enumerate(rows):
        row["anonymous_graph_slots"] = sorted(
            anonymous_match["candidate_to_slots"].get(index, set())
        )
        row["dependency_graph_slots"] = sorted(
            slot_graph_match["candidate_to_slots"].get(index, set())
        )
        row["directed_dependency_graph_slots"] = sorted(
            directed_graph_match["candidate_to_slots"].get(index, set())
        )
        row["anonymous_graph_match"] = bool(row["anonymous_graph_slots"])
        row["dependency_graph_match"] = bool(row["dependency_graph_slots"])
        row["directed_dependency_graph_match"] = bool(
            row["directed_dependency_graph_slots"]
        )
        slot_to_branch = {
            slot["slot_id"]: slot["branch_id"] for slot in slots
        }
        row["anonymous_graph_branches"] = sorted({
            slot_to_branch[slot_id]
            for slot_id in row["anonymous_graph_slots"]
        })
        row["dependency_graph_branches"] = sorted({
            slot_to_branch[slot_id]
            for slot_id in row["dependency_graph_slots"]
        })
        row["directed_dependency_graph_branches"] = sorted({
            slot_to_branch[slot_id]
            for slot_id in row["directed_dependency_graph_slots"]
        })
        row["dependency_match_truncated"] = bool(
            anonymous_match["truncated"]
            or slot_graph_match["truncated"]
            or directed_graph_match["truncated"]
        )
    return rows


def _utility(
    row: dict[str, Any],
    variant: str,
    total: int,
    covered_relations: set[str],
    covered_slots: set[str],
    covered_branches: set[str],
) -> tuple[float, ...]:
    base = 1.0 - row["original_rank"] / max(1, total)
    relation = float(row["relation_match"])
    structure = float(row["structure_match"])
    direction = float(row["direction_match"])
    slot_count = len(
        row["directed_dependency_graph_slots"]
        if variant == "all_factors" else row["dependency_graph_slots"]
    )
    slot_direction_count = len(row["all_factor_slot_direction_matches"])
    new_relation = float(row["relation_match"] and row["relation"] not in covered_relations)
    new_slots = len(set(row["dependency_graph_slots"]) - covered_slots)
    new_slot_directions = len(
        set(row["directed_dependency_graph_slots"]) - covered_slots
    )
    branch_key = {
        "family_structure": "anonymous_graph_branches",
        "relation_slot_branch": "dependency_graph_branches",
        "direction": "directed_dependency_graph_branches",
        "all_factors": "directed_dependency_graph_branches",
    }.get(variant, "branch_matches")
    new_branches = len(set(row[branch_key]) - covered_branches)

    if variant == "baseline":
        return (base,)
    if variant == "family_structure":
        return (float(row["anonymous_graph_match"]), structure, base)
    if variant == "relation_set":
        return (relation, new_relation, base)
    if variant == "relation_slot_branch":
        return (
            float(row["dependency_graph_match"]), float(new_slots),
            float(new_branches), relation, structure, base,
        )
    if variant == "direction":
        return (
            float(row["directed_dependency_graph_match"]),
            float(new_slot_directions), float(new_branches),
            relation, direction, structure, base,
        )
    if variant == "all_factors":
        return (
            float(row["directed_dependency_graph_match"]),
            float(new_slot_directions), float(new_branches),
            float(slot_direction_count > 0),
            float(slot_count > 0), relation, direction, structure,
            float(row["answer_type_match"]), base,
        )
    raise ValueError(f"Unknown oracle variant: {variant}")


def rerank_rows(rows: list[dict[str, Any]], variant: str, greedy_budget: int) -> list[dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
    if variant == "baseline" or not rows:
        return list(rows)

    remaining = list(rows)
    selected = []
    covered_relations: set[str] = set()
    covered_slots: set[str] = set()
    covered_branches: set[str] = set()
    # Novelty can change only until every recoverable relation/slot has been
    # represented once.  Capping dynamic steps by that small factor universe
    # avoids O(pool_size^2) behaviour on the full CWQ test set; the remaining
    # rows receive one deterministic static sort.
    if variant == "relation_set":
        dynamic_universe = {
            row["relation"] for row in rows if row["relation_match"]
        }
    elif variant == "relation_slot_branch":
        dynamic_universe = {
            slot_id for row in rows for slot_id in row["dependency_graph_slots"]
        }
    elif variant == "all_factors":
        dynamic_universe = {
            slot_id for row in rows
            for slot_id in (
                row["directed_dependency_graph_slots"]
                + row["all_factor_slot_direction_matches"]
            )
        }
    elif variant == "direction":
        dynamic_universe = {
            slot_id for row in rows
            for slot_id in row["directed_dependency_graph_slots"]
        }
    else:
        dynamic_universe = set()
    steps = min(max(0, greedy_budget), len(rows), len(dynamic_universe))
    for _ in range(steps):
        best = max(
            remaining,
            key=lambda row: (
                _utility(
                    row, variant, len(rows), covered_relations,
                    covered_slots, covered_branches,
                ),
                -row["original_rank"],
            ),
        )
        remaining.remove(best)
        selected.append(best)
        if best["relation_match"]:
            covered_relations.add(best["relation"])
        # A retrieved triple is one edge and may cover at most one repeated
        # logical slot.  This prevents (r, r) from receiving full slot credit
        # from a single r edge.
        candidate_slots = (
            best["directed_dependency_graph_slots"]
            + best["all_factor_slot_direction_matches"]
            if variant == "all_factors"
            else best["directed_dependency_graph_slots"]
            if variant == "direction"
            else best["dependency_graph_slots"]
        )
        uncovered = sorted(set(candidate_slots) - covered_slots)
        if uncovered:
            covered_slots.add(uncovered[0])
        candidate_branches = (
            best["directed_dependency_graph_branches"]
            if variant in {"direction", "all_factors"}
            else best["dependency_graph_branches"]
        )
        if candidate_branches:
            covered_branches.add(candidate_branches[0])

    # Beyond the largest evaluated budget there is no dynamic coverage benefit.
    # Keep a deterministic static order so every input candidate remains present.
    remaining.sort(
        key=lambda row: (
            _utility(row, variant, len(rows), set(), set(), set()),
            -row["original_rank"],
        ),
        reverse=True,
    )
    return selected + remaining


def _output_triples(rows: list[dict[str, Any]]) -> list[tuple[str, str, str, float]]:
    return [
        (row["head"], row["relation_raw"], row["tail"], row["original_score"])
        for row in rows
    ]


def rerank_sample(
    sample: dict[str, Any],
    factor: dict[str, Any] | None,
    variant: str,
    greedy_budget: int,
    match_state_limit: int = 100_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = annotate_candidates(
        sample, factor, variant, match_state_limit=match_state_limit
    )
    parsed = _factor_usable(factor)
    effective_variant = variant if parsed else "baseline"
    ranked = rerank_rows(rows, effective_variant, greedy_budget)
    output = deepcopy(sample)
    if effective_variant != "baseline":
        key = "scored_triples" if "scored_triples" in sample else "scored_triplets"
        output[key] = _output_triples(ranked)

    before = Counter((row["head"], row["relation_raw"], row["tail"]) for row in rows)
    after = Counter((row["head"], row["relation_raw"], row["tail"]) for row in ranked)
    if before != after:
        raise AssertionError("Reranking changed candidate membership")
    return output, ranked


def rerank_dataset(
    baseline: dict[str, dict[str, Any]],
    factors: dict[str, dict[str, Any]],
    variant: str,
    greedy_budget: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    reranked = {}
    annotated = {}
    for sample_id, sample in baseline.items():
        output, rows = rerank_sample(sample, factors.get(sample_id), variant, greedy_budget)
        reranked[sample_id] = output
        annotated[sample_id] = rows
    if list(reranked) != list(baseline):
        raise AssertionError("Reranking changed sample order or membership")
    return reranked, annotated


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _consistent_branch_metrics(
    top: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    branches: dict[str, set[str]],
    topics: set[str],
    match_state_limit: int,
) -> dict[str, float | None]:
    match = _best_dependency_match(
        top,
        slots,
        topics,
        use_relations=True,
        use_direction=True,
        state_limit=match_state_limit,
    )
    return {
        "consistent_slot_coverage": _safe_div(len(match["matched_slots"]), len(slots)),
        "branch_any_coverage": _safe_div(match["branch_any"], len(branches)),
        "branch_complete_coverage": _safe_div(match["branch_complete"], len(branches)),
        "full_structure_hit": match["full"],
        "structure_match_truncated": float(match["truncated"]),
    }


def sample_retrieval_metrics(
    sample: dict[str, Any],
    factor: dict[str, Any] | None,
    ranked_rows: list[dict[str, Any]],
    budget: int,
    gpt_target_triples: Iterable[Any] = (),
    match_state_limit: int = 100_000,
) -> dict[str, Any]:
    # Mirror the reasoning pipeline: remove duplicate (h,r,t) entries before
    # truncating to the prompt budget.
    top = []
    seen_triples = set()
    for row in ranked_rows:
        triple = (row["head"], row["relation_raw"], row["tail"])
        if triple in seen_triples:
            continue
        seen_triples.add(triple)
        top.append(row)
        if len(top) >= budget:
            break
    predicted_triples = {
        (row["head"], row["relation_raw"], row["tail"]) for row in top
    }
    targets = {
        _triple_tuple(value) for value in sample.get("target_relevant_triples", [])
    }
    gpt_targets = {_triple_tuple(value) for value in gpt_target_triples}
    answers = {str(value) for value in sample.get("a_entity_in_graph", [])}
    predicted_entities = {
        entity for row in top for entity in (row["head"], row["tail"])
    }
    factor = factor or {}
    parsed = _factor_usable(factor)
    relations = set(factor.get("relations", [])) if parsed else set()
    slots = [
        slot for slot in factor.get("relation_slots", [])
        if parsed and slot.get("role", "relation") == "relation"
    ]
    slot_ids = {slot["slot_id"] for slot in slots}
    branch_to_slots = {
        branch["branch_id"]: set(branch["slot_ids"])
        for branch in factor.get("branches", [])
    } if parsed else {}
    covered_relations = {row["relation"] for row in top if row["relation"] in relations}
    def maximum_slot_matching(match_key: str) -> set[str]:
        assigned: dict[str, int] = {}

        def augment(triple_index: int, visited: set[str]) -> bool:
            for slot_id in sorted(top[triple_index][match_key]):
                if slot_id in visited:
                    continue
                visited.add(slot_id)
                if slot_id not in assigned or augment(assigned[slot_id], visited):
                    assigned[slot_id] = triple_index
                    return True
            return False

        for triple_index in range(len(top)):
            augment(triple_index, set())
        return set(assigned)

    covered_slots = maximum_slot_matching("slot_matches")
    covered_slot_directions = maximum_slot_matching("slot_direction_matches")
    branch_presence_any = sum(bool(required & covered_slots) for required in branch_to_slots.values())
    branch_presence_complete = sum(required <= covered_slots for required in branch_to_slots.values())
    topics = {
        str(value) for value in (sample.get("q_entity_in_graph") or sample.get("q_entity") or [])
    }
    consistent = _consistent_branch_metrics(
        top, slots, branch_to_slots, topics, match_state_limit
    )
    return {
        "id": sample.get("id"),
        "budget": budget,
        "parse_status": factor.get("parse_status", "missing"),
        "oracle_eligible": parsed,
        "query_family": factor.get("query_family", "unparsed"),
        "triple_recall": _safe_div(len(targets & predicted_triples), len(targets)),
        "gpt_triple_recall": _safe_div(len(gpt_targets & predicted_triples), len(gpt_targets)),
        "answer_recall": _safe_div(len(answers & predicted_entities), len(answers)),
        "answer_hit": float(bool(answers & predicted_entities)) if answers else None,
        "gold_relation_coverage": _safe_div(len(covered_relations), len(relations)),
        "slot_coverage": _safe_div(len(covered_slots), len(slot_ids)),
        "slot_direction_coverage": _safe_div(len(covered_slot_directions), len(slot_ids)),
        "branch_slot_presence_any_coverage": _safe_div(
            branch_presence_any, len(branch_to_slots)
        ),
        "branch_slot_presence_complete_coverage": _safe_div(
            branch_presence_complete, len(branch_to_slots)
        ),
        **consistent,
        "candidate_count": len(ranked_rows),
        "effective_prompt_k": len(top),
    }


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "triple_recall", "gpt_triple_recall", "answer_recall", "answer_hit", "gold_relation_coverage",
        "slot_coverage", "slot_direction_coverage", "branch_any_coverage",
        "branch_complete_coverage", "consistent_slot_coverage", "full_structure_hit",
        "structure_match_truncated",
        "branch_slot_presence_any_coverage", "branch_slot_presence_complete_coverage",
    )

    def aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        result = {"sample_count": len(group)}
        for metric in metrics:
            result[metric] = _mean(row.get(metric) for row in group)
            result[f"{metric}_denominator"] = sum(row.get(metric) is not None for row in group)
        return result

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["query_family"]].append(row)
        by_status[row["parse_status"]].append(row)
    parsed_rows = [row for row in rows if row["parse_status"] in {"parsed", "partial"}]
    oracle_eligible_rows = [row for row in rows if row.get("oracle_eligible")]
    return {
        "overall": aggregate(rows),
        "parsed_only": aggregate(parsed_rows),
        "oracle_eligible_only": aggregate(oracle_eligible_rows),
        "by_family": {key: aggregate(value) for key, value in sorted(by_family.items())},
        "by_parse_status": {key: aggregate(value) for key, value in sorted(by_status.items())},
    }
