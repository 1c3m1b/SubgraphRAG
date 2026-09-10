from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .io_utils import normalise_question, read_json, sha256_json


FREEBASE_URI = "http://rdf.freebase.com/ns/"
COMMUTATIVE = {"AND", "OR"}
COMPARATIVE = {"LT", "LE", "GT", "GE"}
SUPERLATIVE = {"ARGMAX", "ARGMIN"}
KNOWN_OPERATORS = {
    "AND", "OR", "JOIN", "R", "COUNT", "ARGMAX", "ARGMIN",
    "LT", "LE", "GT", "GE", "TC",
}


def normalize_relation(value: str) -> str:
    value = str(value).strip().strip("<>")
    if value.startswith(FREEBASE_URI):
        value = value[len(FREEBASE_URI) :]
    if value.startswith("ns:"):
        value = value[3:]
    if value.startswith("/"):
        value = value.strip("/").replace("/", ".")
    if value == "a":
        return "type.object.type"
    return value


def _is_entity(value: str) -> bool:
    value = normalize_relation(value)
    return bool(re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", value))


def _is_variable(value: str) -> bool:
    return str(value).startswith("?")


def _is_literal(value: str) -> bool:
    value = str(value)
    return value.startswith(('"', "'")) or bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value))


def _abstract_constant(value: str) -> str:
    value = normalize_relation(value)
    if _is_entity(value):
        return "ENTITY"
    if _is_literal(value):
        return "LITERAL"
    if value.startswith("?"):
        return "VAR"
    return f"CONST:{value}"


def tokenize_sexpr(text: str) -> list[str]:
    return re.findall(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+', text)


def parse_sexpr(text: str) -> Any:
    tokens = tokenize_sexpr(text)
    if not tokens:
        raise ValueError("empty S-expression")
    index = 0

    def parse_one() -> Any:
        nonlocal index
        if index >= len(tokens):
            raise ValueError("unexpected end of S-expression")
        token = tokens[index]
        index += 1
        if token != "(":
            if token == ")":
                raise ValueError("unexpected closing parenthesis")
            return token
        result = []
        while index < len(tokens) and tokens[index] != ")":
            result.append(parse_one())
        if index >= len(tokens):
            raise ValueError("unclosed parenthesis")
        index += 1
        if not result:
            raise ValueError("empty expression")
        return result

    tree = parse_one()
    if index != len(tokens):
        raise ValueError("trailing tokens in S-expression")
    return tree


def _relation_spec(node: Any) -> tuple[str | None, bool]:
    inverse = False
    while isinstance(node, list) and node and str(node[0]).upper() == "R" and len(node) == 2:
        inverse = not inverse
        node = node[1]
    if isinstance(node, str):
        relation = normalize_relation(node)
        if relation and not _is_entity(relation) and not _is_literal(relation):
            return relation, inverse
    return None, inverse


def _canonical_sexpr(node: Any, relation_context: bool = False) -> Any:
    if isinstance(node, str):
        return normalize_relation(node) if relation_context else _abstract_constant(node)
    if not isinstance(node, list) or not node:
        return node
    op = str(node[0]).upper()
    if op == "R" and len(node) == 2:
        relation, inverse = _relation_spec(node)
        if relation is not None:
            return ["R", relation] if inverse else relation
    if op == "JOIN" and len(node) >= 3:
        rel, inverse = _relation_spec(node[1])
        relation_value: Any = ["R", rel] if inverse else rel
        children = [_canonical_sexpr(child) for child in node[2:]]
        return [op, relation_value, *children]
    if op in COMPARATIVE and len(node) >= 3:
        relation, inverse = _relation_spec(node[1])
        relation_value: Any = ["R", relation] if inverse else relation
        return [op, relation_value, *[_canonical_sexpr(child) for child in node[2:]]]
    if op in SUPERLATIVE and len(node) >= 3:
        relation, inverse = _relation_spec(node[2])
        relation_value = ["R", relation] if inverse else relation
        return [op, _canonical_sexpr(node[1]), relation_value,
                *[_canonical_sexpr(child) for child in node[3:]]]
    if op == "TC" and len(node) >= 3:
        relation, inverse = _relation_spec(node[2])
        relation_value = ["R", relation] if inverse else relation
        return [op, _canonical_sexpr(node[1]), relation_value,
                *[_canonical_sexpr(child) for child in node[3:]]]
    children = [_canonical_sexpr(child) for child in node[1:]]
    if op in COMMUTATIVE:
        flattened = []
        for child in children:
            if isinstance(child, list) and child and child[0] == op:
                flattened.extend(child[1:])
            else:
                flattened.append(child)
        children = sorted(flattened, key=lambda item: json.dumps(item, sort_keys=True))
    return [op, *children]


def _walk_ops(node: Any) -> list[str]:
    if not isinstance(node, list) or not node:
        return []
    result = [str(node[0]).upper()]
    for child in node[1:]:
        result.extend(_walk_ops(child))
    return result


def _stable_branches(node: Any) -> list[tuple[str, Any]]:
    if isinstance(node, list) and node and str(node[0]).upper() == "AND":
        children = sorted(
            node[1:],
            key=lambda item: json.dumps(_canonical_sexpr(item), sort_keys=True),
        )
        result = []
        occurrences: defaultdict[str, int] = defaultdict(int)
        for child in children:
            signature = hashlib.sha1(
                json.dumps(_canonical_sexpr(child), sort_keys=True).encode()
            ).hexdigest()[:10]
            occurrence = occurrences[signature]
            occurrences[signature] += 1
            result.append((f"{signature}:{occurrence}", child))
        return result
    signature = hashlib.sha1(json.dumps(_canonical_sexpr(node), sort_keys=True).encode()).hexdigest()[:10]
    return [(signature, node)]


def _assign_topic_relative_fields(slots: list[dict[str, Any]]) -> None:
    """Derive rooted depth/direction from explicit anchor dependencies.

    A global answer-distance heuristic is wrong for unequal multi-anchor
    branches.  This traversal starts from LF topic constants and therefore
    leaves genuinely unanchored or equidistant edges unknown instead of
    inventing a direction.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)
    anchors: set[str] = set()
    for slot in slots:
        if slot.get("role") != "relation":
            continue
        subject = slot.get("subject_dependency")
        obj = slot.get("object_dependency")
        if subject is None or obj is None:
            continue
        adjacency[subject].add(obj)
        adjacency[obj].add(subject)
        if slot.get("subject_dependency_role") == "topic":
            anchors.add(subject)
        if slot.get("object_dependency_role") == "topic":
            anchors.add(obj)

    distances: dict[str, int] = {anchor: 0 for anchor in anchors}
    queue = deque(sorted(anchors))
    while queue:
        node = queue.popleft()
        for neighbour in sorted(adjacency.get(node, ())):
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)

    for slot in slots:
        if slot.get("role") != "relation":
            slot["topic_depth"] = None
            slot["topic_direction"] = "unknown"
            continue
        ds = distances.get(slot.get("subject_dependency"))
        do = distances.get(slot.get("object_dependency"))
        if ds is None or do is None:
            slot["topic_depth"] = None
            slot["topic_direction"] = "unknown"
        elif ds < do:
            slot["topic_depth"] = ds
            slot["topic_direction"] = "forward"
        elif do < ds:
            slot["topic_depth"] = do
            slot["topic_direction"] = "inverse"
        else:
            slot["topic_depth"] = ds
            slot["topic_direction"] = "unknown"


def parse_sexpr_factors(text: str) -> dict[str, Any]:
    tree = parse_sexpr(text)
    canonical = _canonical_sexpr(tree)
    operators = sorted(set(_walk_ops(tree)))
    slots: list[dict[str, Any]] = []
    answer_types: set[str] = set()
    atomic_dependencies: dict[tuple[str, str], str] = {}

    def add_slot(
        relation_node: Any,
        depth: int,
        branch_id: str,
        role: str,
        output_dependency: str,
        output_role: str,
        child_dependency: str,
        child_role: str,
    ) -> None:
        relation, inverse = _relation_spec(relation_node)
        if not relation:
            return
        answer_direction = "inverse" if inverse else "forward"
        subject_dependency = child_dependency if inverse else output_dependency
        object_dependency = output_dependency if inverse else child_dependency
        subject_role = child_role if inverse else output_role
        object_role = output_role if inverse else child_role
        slots.append({
            "relation": relation,
            "answer_direction": answer_direction,
            "topic_direction": "unknown",
            "depth_from_answer": depth,
            "topic_depth": None,
            "branch_id": branch_id,
            "role": role,
            "subject_dependency": subject_dependency,
            "object_dependency": object_dependency,
            "subject_dependency_role": subject_role,
            "object_dependency_role": object_role,
        })

    def child_dependency(
        child: Any,
        branch_id: str,
        depth: int,
        child_index: int,
    ) -> tuple[str, str]:
        payload = json.dumps(
            [branch_id, depth, child_index, _canonical_sexpr(child)],
            sort_keys=True,
        )
        suffix = hashlib.sha1(payload.encode()).hexdigest()[:12]
        if isinstance(child, str):
            if _is_entity(child):
                key = ("topic", normalize_relation(child))
                if key not in atomic_dependencies:
                    atomic_dependencies[key] = f"TOPIC:{len(atomic_dependencies)}"
                return atomic_dependencies[key], "topic"
            if _is_literal(child):
                key = ("literal", str(child))
                if key not in atomic_dependencies:
                    atomic_dependencies[key] = f"LITERAL:{len(atomic_dependencies)}"
                return atomic_dependencies[key], "literal"
            key = ("constant", normalize_relation(child))
            if key not in atomic_dependencies:
                atomic_dependencies[key] = f"CONSTANT:{len(atomic_dependencies)}"
            return atomic_dependencies[key], "constant"
        return f"VARIABLE:{suffix}", "variable"

    def walk(
        node: Any,
        depth: int,
        branch_id: str,
        output_dependency: str,
        output_role: str,
    ) -> None:
        if not isinstance(node, list) or not node:
            return
        op = str(node[0]).upper()
        if op == "JOIN" and len(node) >= 3:
            relation, inverse = _relation_spec(node[1])
            for child_index, child in enumerate(node[2:]):
                child_dep, child_role = child_dependency(
                    child, branch_id, depth, child_index
                )
                if relation:
                    slot_role = (
                        "type" if relation == "type.object.type" else "relation"
                    )
                    add_slot(
                        node[1], depth, branch_id, slot_role,
                        output_dependency, output_role, child_dep, child_role,
                    )
                    if (
                        relation == "type.object.type"
                        and isinstance(child, str)
                        and not _is_variable(child)
                        and not _is_entity(child)
                        and not _is_literal(child)
                    ):
                        answer_types.add(normalize_relation(child))
                walk(child, depth + 1, branch_id, child_dep, child_role)
            return
        if op in SUPERLATIVE and len(node) >= 3:
            walk(node[1], depth, branch_id, output_dependency, output_role)
            operator_branch = hashlib.sha1(f"{branch_id}/{op}/operator".encode()).hexdigest()[:10]
            operator_dep, operator_role = child_dependency(
                node[2], operator_branch, 0, 0
            )
            add_slot(
                node[2], 0, operator_branch, "operator",
                output_dependency, output_role, operator_dep, operator_role,
            )
            return
        if op in COMPARATIVE and len(node) >= 3:
            comparison_dep, comparison_role = child_dependency(
                node[2], branch_id, 0, 0
            )
            add_slot(
                node[1], 0, branch_id, "operator",
                output_dependency, output_role, comparison_dep, comparison_role,
            )
            return
        if op == "TC" and len(node) >= 3:
            walk(node[1], depth, branch_id, output_dependency, output_role)
            operator_branch = hashlib.sha1(f"{branch_id}/TC/operator".encode()).hexdigest()[:10]
            temporal_dep, temporal_role = child_dependency(
                node[2], operator_branch, 0, 0
            )
            add_slot(
                node[2], 0, operator_branch, "operator",
                output_dependency, output_role, temporal_dep, temporal_role,
            )
            return
        if op == "AND":
            for child_branch, child in _stable_branches(node):
                child_id = hashlib.sha1(f"{branch_id}/{child_branch}".encode()).hexdigest()[:10]
                if isinstance(child, str) and not _is_entity(child) and not _is_literal(child):
                    answer_types.add(normalize_relation(child))
                walk(
                    child, depth, child_id, output_dependency, output_role
                )
            return
        for child in node[1:]:
            walk(child, depth, branch_id, output_dependency, output_role)

    root_branch = hashlib.sha1(
        json.dumps(_canonical_sexpr(tree), sort_keys=True).encode()
    ).hexdigest()[:10]
    walk(tree, 0, root_branch, "ANS", "answer")
    _assign_topic_relative_fields(slots)

    slots.sort(key=lambda slot: (
        slot["branch_id"], slot["depth_from_answer"], slot["relation"], slot["answer_direction"]
    ))
    for index, slot in enumerate(slots):
        slot["slot_id"] = f"s{index}"

    result = _finish_factor_dict(
        source_type="sexpr",
        canonical=canonical,
        operators=operators,
        slots=slots,
        answer_types=sorted(answer_types),
    )
    unknown_operators = sorted(set(operators) - KNOWN_OPERATORS)
    result["parse_warnings"] = (
        [f"Unsupported S-expression operators: {', '.join(unknown_operators)}"]
        if unknown_operators else []
    )
    result["unsafe_for_oracle"] = bool(unknown_operators)
    return result


def _strip_sparql_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


TERM = r'(?:\?[A-Za-z_]\w*|[A-Za-z_]\w*:[A-Za-z0-9_.-]+|<[^>]+>|"(?:\\.|[^"\\])*"(?:@[A-Za-z-]+|\^\^\S+)?|[-+]?\d+(?:\.\d+)?)'
SPARQL_TOKEN_RE = re.compile(rf'{TERM}|\ba\b|[.;,{{}}]', re.I)


def _remove_parenthesized_clauses(
    text: str,
    keyword: str,
    preserve_exists: bool = False,
) -> str:
    """Blank FILTER/BIND-like clauses while preserving character boundaries."""
    result = list(text)
    for match in list(re.finditer(rf"\b{keyword}\b\s*\(", text, re.I))[::-1]:
        open_index = text.find("(", match.start())
        if preserve_exists:
            inner_prefix = text[open_index + 1:].lstrip().upper()
            if inner_prefix.startswith("EXISTS") or inner_prefix.startswith("NOT EXISTS"):
                continue
        depth = 0
        quote = None
        escaped = False
        end_index = None
        for index in range(open_index, len(text)):
            char = text[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in {'"', "'"}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end_index = index + 1
                    break
        if end_index is not None:
            result[match.start():end_index] = " " * (end_index - match.start())
    return "".join(result)


def _parenthesized_clause_contents(text: str, keyword: str) -> list[str]:
    """Extract balanced keyword(...) bodies, including nested function calls."""
    contents = []
    for match in re.finditer(rf"\b{keyword}\b\s*\(", text, re.I):
        open_index = text.find("(", match.start())
        depth = 0
        quote = None
        escaped = False
        for index in range(open_index, len(text)):
            char = text[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in {'"', "'"}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    contents.append(text[open_index + 1:index])
                    break
    return contents


def _extract_sparql_triples(text: str) -> tuple[str, list[tuple[str, str, str]], list[str]]:
    cleaned = _strip_sparql_comments(text)
    select_match = re.search(r"SELECT\s+(?:DISTINCT\s+)?(?P<var>\?[A-Za-z_]\w*)", cleaned, re.I)
    answer_var = select_match.group("var") if select_match else "?x"
    body_match = re.search(r"\{(?P<body>.*)\}", cleaned, re.S)
    body = body_match.group("body") if body_match else cleaned
    warnings = []
    upper_body = body.upper()
    if "UNION" in upper_body:
        warnings.append("UNION branches flattened into a union of slots")
    if "OPTIONAL" in upper_body:
        warnings.append("OPTIONAL slots retained without optionality semantics")
    if re.search(r"\b(?:SERVICE|MINUS|VALUES)\b", body, re.I):
        warnings.append("SERVICE/MINUS/VALUES semantics are only partially supported")
    if re.search(r"\b(?:BIND|HAVING|GROUP\s+BY|GRAPH)\b", cleaned, re.I):
        warnings.append("BIND/HAVING/GROUP BY/GRAPH semantics are only partially supported")
    if re.search(
        r"(?:[A-Za-z_]\w*:[A-Za-z0-9_.-]+|<[^>]+>)\s*(?:/|\||\^)\s*"
        r"(?:[A-Za-z_]\w*:[A-Za-z0-9_.-]+|<[^>]+>)",
        body,
    ):
        raise ValueError("SPARQL property paths are not reliably supported")
    if re.search(r"\bFILTER\s*(?:\(|\s)*(?:NOT\s+)?EXISTS\b", body, re.I):
        warnings.append(
            "FILTER EXISTS/NOT EXISTS graph slots retained; negation/existence semantics are partial"
        )

    token_body = _remove_parenthesized_clauses(body, "FILTER", preserve_exists=True)
    token_body = _remove_parenthesized_clauses(token_body, "BIND")
    tokens = [match.group(0) for match in SPARQL_TOKEN_RE.finditer(token_body)]
    triples = []
    index = 0
    subject: str | None = None
    predicate: str | None = None
    while index < len(tokens):
        token = tokens[index]
        if token in {"{", "}"}:
            subject = predicate = None
            index += 1
            continue
        if subject is None:
            if token in {".", ";", ","}:
                index += 1
                continue
            subject = token
            index += 1
        if predicate is None:
            while index < len(tokens) and tokens[index] in {".", "{", "}"}:
                subject = None
                index += 1
            if subject is None or index >= len(tokens):
                continue
            predicate = tokens[index]
            index += 1
        if index >= len(tokens):
            break
        obj = tokens[index]
        if obj in {".", ";", ",", "{", "}"}:
            subject = predicate = None
            index += 1
            continue
        index += 1
        relation = normalize_relation(predicate)
        if not predicate.startswith("?"):
            triples.append((normalize_relation(subject), relation, normalize_relation(obj)))
        elif "variable predicate triple was not converted to a relation slot" not in warnings:
            warnings.append("variable predicate triple was not converted to a relation slot")
        delimiter = tokens[index] if index < len(tokens) and tokens[index] in {".", ";", ","} else None
        if delimiter is not None:
            index += 1
        if delimiter == ";":
            predicate = None
        elif delimiter == ",":
            # Keep both the subject and predicate for an object list.
            pass
        else:
            subject = predicate = None
    # Preserve the first occurrence while eliminating exact duplicates.
    return answer_var, list(dict.fromkeys(triples)), warnings


def _graph_distances(start: str, triples: list[tuple[str, str, str]]) -> dict[str, int]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for subject, _, obj in triples:
        adjacency[subject].add(obj)
        adjacency[obj].add(subject)
    distances = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency.get(node, ()):
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    return distances


def _sparql_branch_ids(
    answer_var: str,
    triples: list[tuple[str, str, str]],
    distances: dict[str, int],
) -> dict[tuple[str, str, str], str]:
    adjacency: dict[str, list[int]] = defaultdict(list)
    for edge_index, (subject, _, obj) in enumerate(triples):
        adjacency[subject].append(edge_index)
        adjacency[obj].append(edge_index)
    labels = _sparql_node_labels(answer_var, triples)
    junctions = {
        node for node, incident in adjacency.items()
        if node == answer_var or not _is_variable(node) or len(incident) != 2
    }
    unvisited = set(range(len(triples)))
    paths: list[list[int]] = []

    def other_end(edge_index: int, node: str) -> str:
        subject, _, obj = triples[edge_index]
        return obj if subject == node else subject

    for start in sorted(junctions, key=lambda node: (labels.get(node, ""), node)):
        for first_edge in sorted(adjacency[start]):
            if first_edge not in unvisited:
                continue
            path = [first_edge]
            unvisited.remove(first_edge)
            previous_edge = first_edge
            current = other_end(first_edge, start)
            while current not in junctions:
                candidates = [
                    edge for edge in adjacency[current]
                    if edge != previous_edge and edge in unvisited
                ]
                if not candidates:
                    break
                next_edge = sorted(candidates)[0]
                path.append(next_edge)
                unvisited.remove(next_edge)
                previous_edge = next_edge
                current = other_end(next_edge, current)
            paths.append(path)
    # A pure variable cycle has no natural endpoint; retain it as one auditable branch.
    while unvisited:
        seed = min(unvisited)
        component = []
        queue = [seed]
        while queue:
            edge_index = queue.pop()
            if edge_index not in unvisited:
                continue
            unvisited.remove(edge_index)
            component.append(edge_index)
            subject, _, obj = triples[edge_index]
            queue.extend(adjacency[subject])
            queue.extend(adjacency[obj])
        paths.append(sorted(component))

    signatures = []
    for path in paths:
        edge_signature = sorted(
            (labels.get(triples[index][0]), triples[index][1], labels.get(triples[index][2]))
            for index in path
        )
        signatures.append((sha256_json(edge_signature), path))
    result: dict[tuple[str, str, str], str] = {}
    occurrences: defaultdict[str, int] = defaultdict(int)
    for signature, path in sorted(signatures, key=lambda item: item[0]):
        occurrence = occurrences[signature]
        occurrences[signature] += 1
        branch_id = hashlib.sha1(f"{signature}:{occurrence}".encode()).hexdigest()[:10]
        for edge_index in path:
            result[triples[edge_index]] = branch_id
    return result


def _sparql_node_labels(
    answer_var: str,
    triples: list[tuple[str, str, str]],
) -> dict[str, str]:
    nodes = {node for triple in triples for node in (triple[0], triple[2])}
    labels = {
        node: "ANS" if node == answer_var else ("VAR" if _is_variable(node) else _abstract_constant(node))
        for node in nodes
    }
    for _ in range(max(2, len(nodes))):
        updated = {}
        for node in nodes:
            incident = []
            for subject, relation, obj in triples:
                if node == subject:
                    incident.append(("out", relation, labels[obj]))
                elif node == obj:
                    incident.append(("in", relation, labels[subject]))
            payload = [labels[node], sorted(incident)]
            updated[node] = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        if updated == labels:
            break
        labels = updated
    return labels


def _sparql_dependency_metadata(
    answer_var: str,
    triples: list[tuple[str, str, str]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Create unique, alpha-renaming-stable dependency IDs and node roles."""
    nodes = {node for triple in triples for node in (triple[0], triple[2])}
    colours = _sparql_node_labels(answer_var, triples)

    def role(node: str) -> str:
        if node == answer_var:
            return "answer"
        if _is_variable(node):
            return "variable"
        if _is_entity(node):
            return "topic"
        if _is_literal(node):
            return "literal"
        return "constant"

    roles = {node: role(node) for node in nodes}
    dependencies = {answer_var: "ANS"}
    prefixes = {
        "variable": "VARIABLE",
        "topic": "TOPIC",
        "literal": "LITERAL",
        "constant": "CONSTANT",
    }
    counters: defaultdict[str, int] = defaultdict(int)
    # Raw names only break exact automorphism ties.  Swapping tied nodes leaves
    # the canonical edge multiset unchanged, so variable/entity renaming is
    # still stable while distinct anchors do not collapse into one node.
    for node in sorted(
        (node for node in nodes if node != answer_var),
        key=lambda value: (roles[value], colours.get(value, ""), value),
    ):
        node_role = roles[node]
        index = counters[node_role]
        counters[node_role] += 1
        dependencies[node] = f"{prefixes[node_role]}:{index}"
    return dependencies, roles


def _sparql_canonical(answer_var: str, triples: list[tuple[str, str, str]]) -> dict[str, Any]:
    labels = _sparql_node_labels(answer_var, triples)
    return {
        "answer": labels.get(answer_var, "ANS"),
        "triples": sorted((labels[s], r, labels[o]) for s, r, o in triples),
    }


def parse_sparql_factors(text: str) -> dict[str, Any]:
    answer_var, triples, parse_warnings = _extract_sparql_triples(text)
    if not triples:
        raise ValueError("no supported SPARQL triple patterns found")
    distances = _graph_distances(answer_var, triples)
    branches = _sparql_branch_ids(answer_var, triples, distances)
    node_dependencies, node_roles = _sparql_dependency_metadata(answer_var, triples)
    operator_variables: set[str] = set()
    for match in re.finditer(
        r"\bORDER\s+BY\s+(?:ASC|DESC)?\s*\(?\s*(\?[A-Za-z_]\w*)",
        text,
        re.I,
    ):
        operator_variables.add(normalize_relation(match.group(1)))
    filter_clauses = _parenthesized_clause_contents(text, "FILTER")
    for clause in filter_clauses:
        if re.search(r"(?<![!])(?:<=|>=|<|>)", clause):
            operator_variables.update(
                normalize_relation(value)
                for value in re.findall(r"\?[A-Za-z_]\w*", clause)
            )
    slots = []
    answer_types = set()
    for subject, relation, obj in triples:
        ds, do = distances.get(subject), distances.get(obj)
        if ds is None or do is None:
            depth = None
            answer_direction = "unknown"
        elif ds <= do:
            depth = ds
            answer_direction = "forward"
        else:
            depth = do
            answer_direction = "inverse"
        if relation == "type.object.type" and subject == answer_var and not _is_variable(obj):
            answer_types.add(obj)
        role = (
            "type" if relation == "type.object.type"
            else "operator" if subject in operator_variables or obj in operator_variables
            else "relation"
        )
        if role == "relation" and (ds is None or do is None):
            warning = "relation slot is disconnected from the selected answer variable"
            if warning not in parse_warnings:
                parse_warnings.append(warning)
        slots.append({
            "relation": relation,
            "answer_direction": answer_direction,
            "topic_direction": "unknown",
            "depth_from_answer": depth,
            "topic_depth": None,
            "branch_id": branches[(subject, relation, obj)],
            "role": role,
            "subject_dependency": node_dependencies.get(subject),
            "object_dependency": node_dependencies.get(obj),
            "subject_dependency_role": node_roles.get(subject),
            "object_dependency_role": node_roles.get(obj),
        })
    _assign_topic_relative_fields(slots)
    slots.sort(key=lambda slot: (
        slot["branch_id"], slot["depth_from_answer"] is None,
        slot["depth_from_answer"] or 0, slot["relation"], slot["answer_direction"]
    ))
    for index, slot in enumerate(slots):
        slot["slot_id"] = f"s{index}"
    operators = []
    upper = text.upper()
    if re.search(r"\bCOUNT\s*\(", upper):
        operators.append("COUNT")
    if "ORDER BY DESC" in upper:
        operators.append("ARGMAX")
    if "ORDER BY ASC" in upper:
        operators.append("ARGMIN")
    if "FILTER" in upper:
        operators.append("FILTER")
        # Preserve semantic comparison operators instead of treating every
        # FILTER as boilerplate.  Language and inequality filters remain tagged
        # as FILTER only.
        for symbol, operator in (("<=", "LE"), (">=", "GE"), ("<", "LT"), (">", "GT")):
            if any(symbol in clause for clause in filter_clauses):
                operators.append(operator)
                break
    if len(set(slot["branch_id"] for slot in slots)) > 1:
        operators.append("AND")
    result = _finish_factor_dict(
        source_type="sparql",
        canonical=_sparql_canonical(answer_var, triples),
        operators=sorted(set(operators)),
        slots=slots,
        answer_types=sorted(answer_types),
    )
    result["parse_warnings"] = parse_warnings
    result["unsafe_for_oracle"] = bool(parse_warnings)
    return result


def _query_family(operators: Iterable[str], slots: list[dict[str, Any]]) -> str:
    operators = set(operators)
    core_slots = [slot for slot in slots if slot.get("role") == "relation"]
    branch_count = len({slot["branch_id"] for slot in core_slots})
    if "COUNT" in operators:
        return "count"
    if operators & SUPERLATIVE:
        return "superlative"
    if operators & COMPARATIVE:
        return "comparative"
    if branch_count > 1:
        branch_sizes = Counter(slot["branch_id"] for slot in core_slots)
        return "mixed" if any(size > 1 for size in branch_sizes.values()) else "conjunction"
    if len(core_slots) > 1:
        return "composition"
    if len(core_slots) == 1:
        return "single"
    return "other"


def _semantic_factor_payload(
    operators: list[str],
    slots: list[dict[str, Any]],
    answer_types: list[str],
) -> dict[str, Any]:
    """Canonical query-graph descriptor shared by SPARQL and S-expressions."""
    node_roles: dict[str, set[str]] = defaultdict(set)
    for slot in slots:
        for side in ("subject", "object"):
            dependency = slot.get(f"{side}_dependency")
            role = slot.get(f"{side}_dependency_role") or "unknown"
            if dependency is not None:
                node_roles[str(dependency)].add(str(role))
    labels = {
        node: "/".join(sorted(roles)) for node, roles in node_roles.items()
    }
    for _ in range(max(2, len(labels))):
        updated = {}
        for node in labels:
            incident = []
            for slot in slots:
                subject = str(slot.get("subject_dependency"))
                obj = str(slot.get("object_dependency"))
                edge = (slot.get("relation"), slot.get("role", "relation"))
                if subject == node and obj in labels:
                    incident.append(("out", *edge, labels[obj]))
                elif obj == node and subject in labels:
                    incident.append(("in", *edge, labels[subject]))
            updated[node] = hashlib.sha1(
                json.dumps([labels[node], sorted(incident)], sort_keys=True).encode()
            ).hexdigest()[:16]
        labels = updated
    edges = sorted(
        (
            labels.get(str(slot.get("subject_dependency")), "UNKNOWN"),
            slot.get("relation"),
            slot.get("role", "relation"),
            labels.get(str(slot.get("object_dependency")), "UNKNOWN"),
        )
        for slot in slots
    )
    # JOIN and R are S-expression syntax, not semantic query operators.
    semantic_operators = sorted(set(operators) - {"JOIN", "R"})
    return {
        "operators": semantic_operators,
        "edges": edges,
        "answer_types": sorted(answer_types),
    }


def _finish_factor_dict(
    source_type: str,
    canonical: Any,
    operators: list[str],
    slots: list[dict[str, Any]],
    answer_types: list[str],
) -> dict[str, Any]:
    relations = sorted({
        slot["relation"] for slot in slots if slot.get("role") == "relation"
    })
    operator_relations = sorted({
        slot["relation"] for slot in slots if slot.get("role") == "operator"
    })
    branches: dict[str, list[str]] = defaultdict(list)
    all_branches: dict[str, list[str]] = defaultdict(list)
    for slot in slots:
        all_branches[slot["branch_id"]].append(slot["slot_id"])
        if slot.get("role") == "relation":
            branches[slot["branch_id"]].append(slot["slot_id"])
    max_hop = max(
        (
            slot["topic_depth"] + 1 for slot in slots
            if slot.get("role") == "relation" and slot["topic_depth"] is not None
        ),
        default=0,
    )
    canonical_payload = {
        "source_type": source_type,
        "canonical": canonical,
        "operators": operators,
    }
    semantic_payload = _semantic_factor_payload(operators, slots, answer_types)
    return {
        "canonical_form": canonical,
        "canonical_signature": sha256_json(canonical_payload),
        "semantic_factor_signature": sha256_json(semantic_payload),
        "operators": operators,
        "query_family": _query_family(operators, slots),
        "relations": relations,
        "operator_relations": operator_relations,
        "relation_slots": slots,
        "branches": [
            {"branch_id": branch_id, "slot_ids": sorted(slot_ids)}
            for branch_id, slot_ids in sorted(branches.items())
        ],
        "all_factor_branches": [
            {"branch_id": branch_id, "slot_ids": sorted(slot_ids)}
            for branch_id, slot_ids in sorted(all_branches.items())
        ],
        "slot_count": len(slots),
        "core_slot_count": sum(slot.get("role") == "relation" for slot in slots),
        "operator_slot_count": sum(slot.get("role") == "operator" for slot in slots),
        "type_slot_count": sum(slot.get("role") == "type" for slot in slots),
        "branch_count": len(branches),
        "max_hop": max_hop,
        "answer_types": answer_types,
    }


def _record_id(record: dict[str, Any]) -> str | None:
    for key in ("id", "ID", "qid", "QuestionId", "question_id"):
        if record.get(key) is not None:
            return str(record[key])
    return None


def _record_question(record: dict[str, Any]) -> str:
    for key in ("question", "RawQuestion", "ProcessedQuestion", "machine_question"):
        if record.get(key):
            return str(record[key])
    return ""


def load_logical_form_records(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """Load official WebQSP/CWQ or ChatKBQA-derived JSON/JSONL files."""
    records: dict[str, dict[str, Any]] = {}
    for path_value in paths:
        path = Path(path_value)
        if path.suffix in {".jsonl", ".gz"} and ".jsonl" in path.name:
            from .io_utils import iter_jsonl
            raw_records = list(iter_jsonl(path))
        else:
            payload = read_json(path)
            if isinstance(payload, dict) and isinstance(payload.get("Questions"), list):
                raw_records = payload["Questions"]
            elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
                raw_records = payload["data"]
            elif isinstance(payload, list):
                raw_records = payload
            else:
                raise ValueError(f"Unsupported logical-form JSON layout: {path}")
        for raw in raw_records:
            if not isinstance(raw, dict):
                continue
            sample_id = _record_id(raw)
            if sample_id is None:
                continue
            candidate_forms = []
            source_answer_types = []
            parses = raw.get("Parses") or raw.get("parses") or []
            if isinstance(parses, list):
                for index, parse in enumerate(parses):
                    if not isinstance(parse, dict):
                        continue
                    for answer in parse.get("Answers", parse.get("answers", [])) or []:
                        if isinstance(answer, dict) and answer.get("AnswerType"):
                            source_answer_types.append(str(answer["AnswerType"]))
                    for key, kind in (("SExpr", "sexpr"), ("sexpr", "sexpr"),
                                      ("Sparql", "sparql"), ("sparql", "sparql")):
                        if parse.get(key):
                            candidate_forms.append({
                                "kind": kind,
                                "text": str(parse[key]),
                                "parse_index": index,
                                "execute_right": bool(parse.get("SExpr_execute_right", False)),
                            })
            for key, kind in (("SExpr", "sexpr"), ("sexpr", "sexpr"),
                              ("s_expression", "sexpr"), ("Sparql", "sparql"),
                              ("sparql", "sparql"), ("sparql_query", "sparql")):
                if raw.get(key):
                    candidate_forms.append({
                        "kind": kind,
                        "text": str(raw[key]),
                        "parse_index": -1,
                        "execute_right": bool(raw.get("SExpr_execute_right", False)),
                    })
            # Deduplicate without losing the source preference.
            seen = set()
            unique_forms = []
            for form in candidate_forms:
                key = (form["kind"], form["text"])
                if key not in seen:
                    seen.add(key)
                    unique_forms.append(form)
            incoming_question = _record_question(raw)
            if sample_id not in records:
                records[sample_id] = {
                    "id": sample_id,
                    "question": incoming_question,
                    "source_paths": [str(path)],
                    "source_family": raw.get("compositionality_type") or raw.get("query_type"),
                    "source_answer_type": raw.get("answer_type"),
                    "candidate_forms": [],
                    "source_conflicts": [],
                }
            record = records[sample_id]
            if source_answer_types:
                existing_types = record.get("source_answer_type")
                if existing_types is None:
                    record["source_answer_type"] = sorted(set(source_answer_types))
                else:
                    if not isinstance(existing_types, list):
                        existing_types = [existing_types]
                    record["source_answer_type"] = sorted({
                        *map(str, existing_types), *source_answer_types,
                    })
            if str(path) not in record["source_paths"]:
                record["source_paths"].append(str(path))
            if (
                incoming_question
                and record.get("question")
                and normalise_question(incoming_question) != normalise_question(record["question"])
            ):
                record["source_conflicts"].append({
                    "field": "question",
                    "source_path": str(path),
                    "value": incoming_question,
                })
            elif incoming_question and not record.get("question"):
                record["question"] = incoming_question
            existing_forms = {
                (form["kind"], form["text"]) for form in record["candidate_forms"]
            }
            for form in unique_forms:
                key = (form["kind"], form["text"])
                if key not in existing_forms:
                    enriched = dict(form)
                    enriched["source_path"] = str(path)
                    record["candidate_forms"].append(enriched)
                    existing_forms.add(key)
    return records


def extract_factor_record(
    sample_id: str,
    baseline_sample: dict[str, Any],
    source_record: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "id": sample_id,
        "question": baseline_sample.get("question", ""),
        "parse_status": "missing",
        "parse_errors": [],
        "parse_warnings": [],
        "oracle_eligible": False,
        "alignment": {
            "source_present": source_record is not None,
            "question_match": None,
        },
        "logical_form_type": None,
        "raw_logical_form": None,
        "source_path": None,
        "source_family": None,
        "annotation_answer_types": [],
    }
    if source_record is None:
        return base
    base["source_path"] = source_record.get("source_paths", [])
    base["source_family"] = source_record.get("source_family")
    source_question = source_record.get("question", "")
    base["alignment"]["question_match"] = (
        not source_question
        or normalise_question(source_question) == normalise_question(base["question"])
    )
    base["alignment"]["source_conflicts"] = source_record.get("source_conflicts", [])
    forms = sorted(
        source_record.get("candidate_forms", []),
        key=lambda form: (
            not form.get("execute_right", False),
            form.get("kind") != "sexpr",
            len(form.get("text", "")),
            form.get("text", ""),
            form.get("parse_index", 10**9),
        ),
    )
    parsed_candidates = []
    for form in forms:
        try:
            if form["kind"] == "sexpr":
                factors = parse_sexpr_factors(form["text"])
            else:
                factors = parse_sparql_factors(form["text"])
            parsed_candidates.append((form, factors))
        except Exception as exc:  # retain all failures for audit instead of dropping samples
            base["parse_errors"].append(f"{form.get('kind')}: {type(exc).__name__}: {exc}")
    if not parsed_candidates:
        base["parse_status"] = "unparsed" if forms else "missing"
        return base

    form, factors = parsed_candidates[0]
    base["parse_warnings"] = factors.get("parse_warnings", [])
    base.update(deepcopy(factors))
    if base["alignment"]["question_match"] is False or source_record.get("source_conflicts"):
        # Keep all extracted information for audit, but prevent an LF from a
        # mismatched dataset snapshot from influencing the oracle ranking.
        base["parse_status"] = "alignment_mismatch"
        base["oracle_eligible"] = False
    else:
        base["parse_status"] = (
            "parsed"
            if not base["parse_errors"] and not base["parse_warnings"]
            else "partial"
        )
        base["oracle_eligible"] = not factors.get("unsafe_for_oracle", False)
    base["logical_form_type"] = form["kind"]
    base["raw_logical_form"] = form["text"]
    base["alternative_count"] = len(parsed_candidates)
    base["alternative_signatures"] = sorted({item[1]["canonical_signature"] for item in parsed_candidates})
    base["alternative_semantic_signatures"] = sorted({
        item[1]["semantic_factor_signature"] for item in parsed_candidates
    })
    base["alternatives"] = [
        {
            "logical_form_type": candidate_form["kind"],
            "raw_logical_form": candidate_form["text"],
            "canonical_signature": candidate_factors["canonical_signature"],
            "semantic_factor_signature": candidate_factors["semantic_factor_signature"],
            "query_family": candidate_factors["query_family"],
            "operators": candidate_factors["operators"],
            "relations": candidate_factors["relations"],
            "operator_relations": candidate_factors.get("operator_relations", []),
            "relation_slots": candidate_factors["relation_slots"],
            "branches": candidate_factors["branches"],
            "all_factor_branches": candidate_factors.get("all_factor_branches", []),
            "answer_types": candidate_factors["answer_types"],
            "parse_warnings": candidate_factors.get("parse_warnings", []),
            "unsafe_for_oracle": candidate_factors.get("unsafe_for_oracle", False),
        }
        for candidate_form, candidate_factors in parsed_candidates
    ]
    base["cross_form_consistent"] = len({
        item[1]["semantic_factor_signature"] for item in parsed_candidates
    }) <= 1
    source_answer_type = source_record.get("source_answer_type")
    if source_answer_type:
        base["annotation_answer_types"] = (
            [str(value) for value in source_answer_type]
            if isinstance(source_answer_type, list)
            else [str(source_answer_type)]
        )
    return base
