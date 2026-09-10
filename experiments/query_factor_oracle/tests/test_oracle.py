from __future__ import annotations

import copy
import unittest

from experiments.query_factor_oracle.qforacle.logical_form import parse_sexpr_factors
from experiments.query_factor_oracle.qforacle.oracle import (
    annotate_candidates,
    rerank_sample,
    sample_retrieval_metrics,
)


def factor(expression: str) -> dict:
    result = parse_sexpr_factors(expression)
    result["parse_status"] = "parsed"
    return result


class OracleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {
            "question": "synthetic",
            "q_entity": ["topic"],
            "q_entity_in_graph": ["topic"],
            "a_entity": ["answer"],
            "a_entity_in_graph": ["answer"],
            "target_relevant_triples": [("topic", "gold.r", "answer")],
            "scored_triples": [
                ("topic", "noise.r", "noise", 0.9),
                ("topic", "gold.r", "answer", 0.8),
                ("other", "gold.r", "far", 0.7),
            ],
        }

    def test_relation_set_promotes_gold_relation_and_preserves_scores(self) -> None:
        output, _ = rerank_sample(self.sample, factor("(JOIN (R gold.r) m.a)"), "relation_set", 3)
        self.assertEqual(output["scored_triples"][0][:3], ("topic", "gold.r", "answer"))
        self.assertEqual(output["scored_triples"][0][3], 0.8)
        self.assertCountEqual(
            [item[:3] for item in output["scored_triples"]],
            [item[:3] for item in self.sample["scored_triples"]],
        )

    def test_direction_arm_is_srd_increment_over_undirected_slots(self) -> None:
        gold = factor("(JOIN (R gold.r) m.a)")
        directional = {
            **self.sample,
            "q_entity": ["topic"],
            "q_entity_in_graph": ["topic"],
            "scored_triples": [
                ("wrong", "gold.r", "topic", 0.9),
                ("topic", "gold.r", "answer", 0.8),
            ],
        }
        undirected, _ = rerank_sample(
            directional, gold, "relation_slot_branch", 1
        )
        directed, _ = rerank_sample(directional, gold, "direction", 1)
        self.assertEqual(
            undirected["scored_triples"][0][:3],
            ("wrong", "gold.r", "topic"),
        )
        self.assertEqual(
            directed["scored_triples"][0][:3],
            ("topic", "gold.r", "answer"),
        )

    def test_unparsed_is_exact_baseline(self) -> None:
        output, _ = rerank_sample(self.sample, {"parse_status": "unparsed"}, "all_factors", 3)
        self.assertEqual(output, self.sample)

    def test_semantically_unsafe_partial_is_neutral(self) -> None:
        unsafe = factor("(JOIN (R gold.r) m.a)")
        unsafe["parse_status"] = "partial"
        unsafe["oracle_eligible"] = False
        output, _ = rerank_sample(self.sample, unsafe, "all_factors", 3)
        self.assertEqual(output, self.sample)

    def test_eval_labels_do_not_influence_ranking(self) -> None:
        gold = factor("(JOIN (R gold.r) m.a)")
        first, _ = rerank_sample(self.sample, gold, "all_factors", 3)
        changed = copy.deepcopy(self.sample)
        changed["a_entity_in_graph"] = ["unrelated"]
        changed["target_relevant_triples"] = [("x", "y", "z")]
        second, _ = rerank_sample(changed, gold, "all_factors", 3)
        self.assertEqual(first["scored_triples"], second["scored_triples"])

    def test_duplicate_relation_slots_need_distinct_triples(self) -> None:
        repeated = factor("(JOIN r (JOIN r m.a))")
        one_edge = {
            **self.sample,
            "scored_triples": [("topic", "r", "answer", 0.9)],
            "target_relevant_triples": [],
        }
        rows = annotate_candidates(one_edge, repeated)
        metrics = sample_retrieval_metrics(one_edge, repeated, rows, 100)
        self.assertEqual(repeated["slot_count"], 2)
        self.assertEqual(metrics["gold_relation_coverage"], 1.0)
        self.assertEqual(metrics["slot_coverage"], 0.5)

    def test_prompt_metric_deduplicates_before_budget(self) -> None:
        duplicated = copy.deepcopy(self.sample)
        duplicated["scored_triples"] = [
            ("topic", "noise.r", "noise", 0.9),
            ("topic", "noise.r", "noise", 0.8),
            ("topic", "gold.r", "answer", 0.7),
        ]
        gold = factor("(JOIN (R gold.r) m.a)")
        rows = annotate_candidates(duplicated, gold)
        metrics = sample_retrieval_metrics(duplicated, gold, rows, 2)
        self.assertEqual(metrics["effective_prompt_k"], 2)
        self.assertEqual(metrics["answer_recall"], 1.0)

    def test_full_structure_requires_conjunction_branches_to_share_terminal(self) -> None:
        gold = factor("(AND (JOIN (R r.a) m.t1) (JOIN (R r.b) m.t2))")
        disconnected = {
            **self.sample,
            "q_entity": ["m.t1", "m.t2"],
            "q_entity_in_graph": ["m.t1", "m.t2"],
            "target_relevant_triples": [],
            "scored_triples": [
                ("m.t1", "r.a", "x1", 0.9),
                ("m.t2", "r.b", "x2", 0.8),
            ],
        }
        rows = annotate_candidates(disconnected, gold)
        metrics = sample_retrieval_metrics(disconnected, gold, rows, 2)
        self.assertEqual(metrics["branch_slot_presence_complete_coverage"], 1.0)
        # Presence-only coverage is full, but a single consistent query-graph
        # binding can complete only one of the disconnected branches.
        self.assertEqual(metrics["branch_complete_coverage"], 0.5)
        self.assertEqual(metrics["full_structure_hit"], 0.0)

        connected = copy.deepcopy(disconnected)
        connected["scored_triples"][1] = ("m.t2", "r.b", "x1", 0.8)
        rows = annotate_candidates(connected, gold)
        metrics = sample_retrieval_metrics(connected, gold, rows, 2)
        self.assertEqual(metrics["full_structure_hit"], 1.0)

    def test_intermediate_junction_query_graph_match_and_rerank(self) -> None:
        from experiments.query_factor_oracle.qforacle.logical_form import parse_sparql_factors

        gold = parse_sparql_factors(
            "SELECT DISTINCT ?x WHERE { ?x ns:r.one ?y . "
            "?y ns:r.two ns:m.a . ?y ns:r.three ns:m.b . }"
        )
        gold["parse_status"] = "parsed"
        sample = {
            **self.sample,
            "q_entity": ["m.a", "m.b"],
            "q_entity_in_graph": ["m.a", "m.b"],
            "target_relevant_triples": [],
            "scored_triples": [
                # Higher-scoring relation matches do not share one junction.
                ("bad_ans", "r.one", "bad_j1", 0.99),
                ("bad_j1", "r.two", "m.a", 0.98),
                ("bad_j2", "r.three", "m.b", 0.97),
                # Lower-scoring edges form the complete gold dependency graph.
                ("good_ans", "r.one", "good_j", 0.60),
                ("good_j", "r.two", "m.a", 0.59),
                ("good_j", "r.three", "m.b", 0.58),
            ],
        }
        baseline_rows = annotate_candidates(sample, gold)
        baseline_metrics = sample_retrieval_metrics(sample, gold, baseline_rows, 3)
        self.assertEqual(baseline_metrics["full_structure_hit"], 0.0)

        _, ranked = rerank_sample(sample, gold, "all_factors", 3)
        reranked_metrics = sample_retrieval_metrics(sample, gold, ranked, 3)
        self.assertEqual(reranked_metrics["consistent_slot_coverage"], 1.0)
        self.assertEqual(reranked_metrics["branch_complete_coverage"], 1.0)
        self.assertEqual(reranked_metrics["full_structure_hit"], 1.0)
        self.assertEqual(
            {(row["head"], row["relation_raw"], row["tail"]) for row in ranked[:3]},
            {
                ("good_ans", "r.one", "good_j"),
                ("good_j", "r.two", "m.a"),
                ("good_j", "r.three", "m.b"),
            },
        )


if __name__ == "__main__":
    unittest.main()
