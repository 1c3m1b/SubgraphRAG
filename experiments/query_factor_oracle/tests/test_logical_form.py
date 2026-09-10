from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.query_factor_oracle.qforacle.logical_form import (
    extract_factor_record,
    load_logical_form_records,
    normalize_relation,
    parse_sexpr_factors,
    parse_sparql_factors,
)


class LogicalFormTest(unittest.TestCase):
    def test_and_branch_order_and_entity_identity_are_canonical(self) -> None:
        left = parse_sexpr_factors("(AND (JOIN people.person.place_of_birth m.a) (JOIN (R people.person.parents) m.b))")
        right = parse_sexpr_factors("(AND (JOIN (R people.person.parents) m.x) (JOIN people.person.place_of_birth m.y))")
        self.assertEqual(left["canonical_signature"], right["canonical_signature"])
        self.assertEqual(left["query_family"], "conjunction")

    def test_inverse_and_double_inverse(self) -> None:
        inverse = parse_sexpr_factors("(JOIN (R people.person.parents) m.a)")
        double = parse_sexpr_factors("(JOIN (R (R people.person.parents)) m.a)")
        self.assertEqual(inverse["relation_slots"][0]["topic_direction"], "forward")
        self.assertEqual(double["relation_slots"][0]["topic_direction"], "inverse")

    def test_sparql_variable_names_and_triple_order_are_canonical(self) -> None:
        first = parse_sparql_factors(
            "SELECT DISTINCT ?x WHERE { ?x ns:r.one ?y . ?y ns:r.two ns:m.a . }"
        )
        second = parse_sparql_factors(
            "SELECT DISTINCT ?answer WHERE { ?z ns:r.two ns:m.other . ?answer ns:r.one ?z . }"
        )
        self.assertEqual(first["canonical_signature"], second["canonical_signature"])
        self.assertEqual(first["max_hop"], 2)

    def test_sparql_semicolon_and_comma_shorthand(self) -> None:
        factors = parse_sparql_factors(
            "SELECT DISTINCT ?x WHERE { ?x ns:r.one ?y ; ns:r.two ns:m.a , ns:m.b . }"
        )
        self.assertEqual(factors["relations"], ["r.one", "r.two"])
        self.assertEqual(factors["slot_count"], 3)

    def test_sparql_intermediate_junction_splits_maximal_path_branches(self) -> None:
        factors = parse_sparql_factors(
            "SELECT DISTINCT ?x WHERE { ?x ns:r.one ?y . "
            "?y ns:r.two ns:m.a . ?y ns:r.three ns:m.b . }"
        )
        self.assertEqual(factors["query_family"], "conjunction")
        self.assertEqual(factors["branch_count"], 3)
        dependencies = [
            {slot["subject_dependency"], slot["object_dependency"]}
            for slot in factors["relation_slots"]
        ]
        self.assertTrue(set.intersection(*dependencies))

    def test_sparql_unequal_anchor_branches_have_local_topic_depths(self) -> None:
        factors = parse_sparql_factors(
            "SELECT DISTINCT ?x WHERE { ?x ns:r.short ns:m.s . "
            "?x ns:r.long1 ?y . ?y ns:r.long2 ns:m.t . }"
        )
        depths = {
            slot["relation"]: slot["topic_depth"]
            for slot in factors["relation_slots"]
        }
        self.assertEqual(depths, {"r.short": 0, "r.long1": 1, "r.long2": 0})

    def test_cross_format_semantic_signature_checks_direction(self) -> None:
        sexpr = parse_sexpr_factors("(JOIN r.one m.a)")
        equivalent = parse_sparql_factors(
            "SELECT DISTINCT ?answer WHERE { ?answer ns:r.one ns:m.other . }"
        )
        opposite = parse_sparql_factors(
            "SELECT DISTINCT ?answer WHERE { ns:m.other ns:r.one ?answer . }"
        )
        self.assertEqual(
            sexpr["semantic_factor_signature"],
            equivalent["semantic_factor_signature"],
        )
        self.assertNotEqual(
            sexpr["semantic_factor_signature"],
            opposite["semantic_factor_signature"],
        )

    def test_unsupported_property_path_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "property paths"):
            parse_sparql_factors(
                "SELECT DISTINCT ?x WHERE { ?x ns:r.one/ns:r.two ns:m.a . }"
            )

    def test_filter_exists_keeps_graph_slots_and_marks_partial_semantics(self) -> None:
        factors = parse_sparql_factors(
            "SELECT DISTINCT ?x WHERE { ?x ns:r.one ns:m.a . "
            "FILTER(EXISTS { ?x ns:r.required ?y . }) }"
        )
        self.assertEqual(factors["relations"], ["r.one", "r.required"])
        self.assertTrue(any("EXISTS" in warning for warning in factors["parse_warnings"]))

    def test_relation_notation_normalization(self) -> None:
        expected = "people.person.place_of_birth"
        self.assertEqual(normalize_relation("ns:people.person.place_of_birth"), expected)
        self.assertEqual(normalize_relation("http://rdf.freebase.com/ns/people.person.place_of_birth"), expected)
        self.assertEqual(normalize_relation("/people/person/place_of_birth"), expected)

    def test_explicit_answer_type_and_comparative_are_retained(self) -> None:
        typed = parse_sexpr_factors("(JOIN type.object.type people.person)")
        self.assertEqual(typed["answer_types"], ["people.person"])
        compared = parse_sparql_factors(
            "SELECT DISTINCT ?x WHERE { ?x ns:people.person.age ?age . FILTER (?age > 30) }"
        )
        self.assertIn("GT", compared["operators"])
        self.assertEqual(compared["query_family"], "comparative")

    def test_sexpr_operator_relations_and_typed_and_are_retained(self) -> None:
        ranked = parse_sexpr_factors("(ARGMAX (JOIN r.one m.a) r.date)")
        self.assertEqual(ranked["query_family"], "superlative")
        self.assertEqual(ranked["relations"], ["r.one"])
        self.assertEqual(ranked["operator_relations"], ["r.date"])
        self.assertEqual(
            {slot["role"] for slot in ranked["relation_slots"]},
            {"relation", "operator"},
        )
        compared = parse_sexpr_factors("(GT r.height 100)")
        self.assertEqual(compared["query_family"], "comparative")
        self.assertEqual(compared["operator_relations"], ["r.height"])
        typed = parse_sexpr_factors("(AND people.person (JOIN r.one m.a))")
        self.assertEqual(typed["answer_types"], ["people.person"])
        self.assertEqual(typed["query_family"], "single")

    def test_sparql_and_sexpr_operator_relation_universes_match(self) -> None:
        sexpr = parse_sexpr_factors("(ARGMAX (JOIN r.one m.a) r.date)")
        sparql = parse_sparql_factors(
            "SELECT DISTINCT ?x WHERE { ?x ns:r.one ns:m.a . ?x ns:r.date ?d . } "
            "ORDER BY DESC(?d) LIMIT 1"
        )
        self.assertEqual(sexpr["relations"], sparql["relations"])
        self.assertEqual(sexpr["operator_relations"], sparql["operator_relations"])

    def test_nested_filter_function_and_spaced_count_are_parsed(self) -> None:
        compared = parse_sparql_factors(
            'SELECT DISTINCT ?x WHERE { ?x ns:r.date ?d . '
            'FILTER (xsd:datetime(?d) >= "2000-01-01"^^xsd:dateTime) }'
        )
        self.assertIn("GE", compared["operators"])
        self.assertEqual(compared["query_family"], "comparative")
        self.assertEqual(compared["relations"], [])
        self.assertEqual(compared["operator_relations"], ["r.date"])
        counted = parse_sparql_factors(
            "SELECT (COUNT (?x) AS ?count) WHERE { ?x ns:r.one ns:m.a . }"
        )
        self.assertEqual(counted["query_family"], "count")

    def test_loader_merges_sources_and_mismatch_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "a.json"
            second = Path(temp) / "b.json"
            first.write_text(json.dumps([{
                "id": "q1", "question": "Who?", "SExpr": "(JOIN r m.a)"
            }]), encoding="utf-8")
            second.write_text(json.dumps([{
                "id": "q1", "question": "A different question?",
                "sparql": "SELECT DISTINCT ?x WHERE { ?x ns:r ns:m.a . }"
            }]), encoding="utf-8")
            sources = load_logical_form_records([first, second])
            self.assertEqual(len(sources["q1"]["candidate_forms"]), 2)
            record = extract_factor_record(
                "q1", {"question": "Who?"}, sources["q1"]
            )
            self.assertEqual(record["parse_status"], "alignment_mismatch")
            self.assertTrue(record["alignment"]["source_conflicts"])

    def test_missing_and_unparseable_samples_are_not_dropped(self) -> None:
        missing = extract_factor_record("q", {"question": "Q"}, None)
        self.assertEqual(missing["parse_status"], "missing")
        source = {
            "question": "Q", "source_paths": ["synthetic"], "source_conflicts": [],
            "candidate_forms": [{"kind": "sexpr", "text": "(", "execute_right": False}],
        }
        bad = extract_factor_record("q", {"question": "Q"}, source)
        self.assertEqual(bad["parse_status"], "unparsed")
        self.assertTrue(bad["parse_errors"])

    def test_annotation_answer_type_is_audited_but_not_used_as_gold_factor(self) -> None:
        source = {
            "question": "Q", "source_paths": ["synthetic"], "source_conflicts": [],
            "source_answer_type": "Entity",
            "candidate_forms": [{
                "kind": "sexpr", "text": "(JOIN r m.a)", "execute_right": True,
            }],
        }
        record = extract_factor_record("q", {"question": "Q"}, source)
        self.assertEqual(record["answer_types"], [])
        self.assertEqual(record["annotation_answer_types"], ["Entity"])


if __name__ == "__main__":
    unittest.main()
