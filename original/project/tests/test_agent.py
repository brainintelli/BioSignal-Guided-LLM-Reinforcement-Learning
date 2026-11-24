"""Behavior tests: rejecting illegal actions and preserving model attribution."""
import copy
import json
import unittest
import threading
import urllib.request
from http.server import HTTPServer

from agent.broker import DecisionEngine, handler_for, validate_observation, validate_action


def observation():
    return {
        "schema_version": 1, "tick": 12, "day": 1, "phase": "day",
        "player": {"health": 90, "hunger": 20, "thirst": 60, "energy": 80, "x": 0, "y": 0},
        "inventory": {"wood": 0, "stone": 0, "berries": 2},
        "nearby": [], "threats": [], "camp": {"exists": False, "x": 0, "y": 0},
        "allowed_actions": [{"verb": "wait"}, {"verb": "eat"}],
    }


class ProtocolTests(unittest.TestCase):
    def test_unknown_command_rejected_even_if_engine_lists_it(self):
        obs = observation()
        obs["allowed_actions"].append({"verb": "exec", "target_id": "calc.exe"})
        with self.assertRaises(ValueError):
            validate_observation(obs)

    def test_action_cannot_invent_target(self):
        obs = observation()
        obs["allowed_actions"].append({"verb": "move", "target_id": "tree_1"})
        with self.assertRaises(ValueError):
            validate_action({"verb": "move", "target_id": "tree_2"}, obs)

    def test_extra_keys_are_not_silently_discarded(self):
        with self.assertRaises(ValueError):
            validate_action({"verb": "eat", "command": "anything"}, observation())

    def test_wait_required_for_safe_failure(self):
        obs = observation()
        obs["allowed_actions"] = [{"verb": "eat"}]
        with self.assertRaises(ValueError):
            validate_observation(obs)

    def test_non_finite_stats_rejected(self):
        obs = observation()
        obs["player"]["health"] = float("nan")
        with self.assertRaises(ValueError):
            validate_observation(obs)


class DecisionTests(unittest.TestCase):
    def test_constraint_assistance_is_explicit_in_decision_metadata(self):
        result = DecisionEngine(model="test", completion=lambda _: '{"verb":"eat"}',
                                constrained=True).decide(observation())
        self.assertTrue(result["assisted_decoding"])
        self.assertEqual(result["source"], "model")
        self.assertIn("JSON", result["reason"])
        self.assertEqual(len(result["action_schema_sha256"]), 64)

    def test_baseline_is_never_marked_as_model(self):
        result = DecisionEngine().decide(observation())
        self.assertEqual(result["source"], "baseline")
        self.assertEqual(result["action"], {"verb": "eat"})
        self.assertIsNone(result["raw_valid"])

    def test_valid_model_response_preserved(self):
        result = DecisionEngine(model="test", completion=lambda _: '{"verb":"eat"}').decide(observation())
        self.assertEqual(result["source"], "model")
        self.assertTrue(result["raw_valid"])
        self.assertEqual(result["action"], {"verb": "eat"})

    def test_illegal_model_output_falls_back_to_wait_and_counts_invalid(self):
        result = DecisionEngine(model="test", completion=lambda _: '{"verb":"spawn_food"}').decide(observation())
        self.assertEqual(result["source"], "fallback")
        self.assertFalse(result["raw_valid"])
        self.assertEqual(result["action"], {"verb": "wait"})

    def test_timeout_is_not_a_valid_model_action(self):
        def timeout(_):
            raise TimeoutError("test timeout")
        result = DecisionEngine(model="test", completion=timeout).decide(observation())
        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["error_type"], "TimeoutError")
        self.assertFalse(result["raw_valid"])

    def test_markdown_wrapping_counts_as_invalid_raw_output(self):
        result = DecisionEngine(model="test", completion=lambda _: '```json\n{"verb":"eat"}\n```').decide(observation())
        self.assertFalse(result["raw_valid"])

    def test_caller_observation_is_not_mutated(self):
        obs = observation()
        original = copy.deepcopy(obs)
        DecisionEngine().decide(obs)
        self.assertEqual(obs, original)


class EvaluationTests(unittest.TestCase):
    def test_baseline_report_has_no_model_accuracy(self):
        from agent.evaluate import evaluate
        report = evaluate(DecisionEngine(), repeats=1)
        self.assertEqual(report["model_calls"], 0)
        self.assertIsNone(report["model_correct_rate"])
        self.assertFalse(report["long_horizon_tested"])

    def test_invalid_output_never_gets_credit_for_fallback(self):
        from agent.evaluate import evaluate
        engine = DecisionEngine(model="bad", completion=lambda _: '{"verb":"teleport"}')
        report = evaluate(engine, repeats=1)
        self.assertEqual(report["raw_valid_rate"], 0)
        self.assertEqual(report["model_correct_rate"], 0)
        self.assertEqual(report["fallback_rate"], 1)


class HttpIntegrationTests(unittest.TestCase):
    def test_real_http_health_and_decision_roundtrip(self):
        server = HTTPServer(("127.0.0.1", 0), handler_for(DecisionEngine()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = "http://127.0.0.1:" + str(server.server_address[1])
        try:
            with urllib.request.urlopen(endpoint + "/health", timeout=2) as response:
                self.assertEqual(json.load(response)["mode"], "baseline")
            request = urllib.request.Request(endpoint + "/decide", json.dumps(observation()).encode(),
                                             {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=2) as response:
                result = json.load(response)
            self.assertEqual(result["action"], {"verb": "eat"})
            self.assertEqual(result["tick"], 12)
            self.assertEqual(result["source"], "baseline")
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()


if __name__ == "__main__":
    unittest.main()
