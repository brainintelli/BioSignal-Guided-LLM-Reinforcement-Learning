"""Release binding for M8. Recompute all frozen evidence; never trust a gate bit."""
from __future__ import annotations
import json
from pathlib import Path

from agent.m8_identity import BASE_WEIGHT_SHA, sha
from agent.m8_policy_identity import PARENT_ACTOR_SHA256
from tools.compare_m8_survival import build_result

CONTRACT = "open_world_semantic_v4"
LAYOUT = "fixed_batch4_left_pad_v1"
SHAPE = [4, 1536]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def verify_release(*, verification, project, checkpoint, model, actor_sha256,
                   policy_metadata, execution_identity):
    project, checkpoint, model = map(lambda p: Path(p).resolve(), (project, checkpoint, model))
    report = json.loads(Path(verification).read_text(encoding="utf-8"))
    rebuilt = build_result(report["publication_request"], project)
    if canonical(report) != canonical(rebuilt):
        raise ValueError("Complete publication result does not reproduce")
    if (report.get("protocol") != "m8_reliable_v1" or report.get("phase") != "test"
            or report.get("decision") != "accept" or report.get("release_authorized") is not True
            or report.get("evidence_complete") is not True):
        raise ValueError("M8 has no complete accepted held-out release")
    config = report["config"]
    if (Path(config["project"]).resolve() != project or Path(config["model"]).resolve() != model
            or Path(config["output"]).resolve() / report["selected_checkpoint"] != checkpoint
            or Path(report["publication_request"]["training_run"]).resolve() != checkpoint.parent
            or config.get("profile") != "open1" or config.get("contract") != CONTRACT
            or config.get("max_tokens") != 1536 or config.get("temperature") != 1):
        raise ValueError("Serving paths/config differ from registered selected checkpoint")
    if (report.get("selected_actor_sha256") != actor_sha256
            or report.get("parent_actor_sha256") != PARENT_ACTOR_SHA256
            or canonical(execution_identity) != canonical(report["execution_identity"])
            or execution_identity.get("base_weight_sha256") != BASE_WEIGHT_SHA):
        raise ValueError("Serving actor/runtime/base/tokenizer/parent identity differs")
    saved = json.loads((checkpoint / "policy.json").read_text(encoding="utf-8"))
    if canonical(saved) != canonical(policy_metadata):
        raise ValueError("Loaded policy metadata differs from selected artifact")
    if (policy_metadata.get("actor_sha256") != actor_sha256 or policy_metadata.get("contract") != CONTRACT
            or policy_metadata.get("reference_actor_sha256") != PARENT_ACTOR_SHA256
            or policy_metadata.get("inference_layout") != LAYOUT
            or policy_metadata.get("physical_forward_shape") != SHAPE
            or policy_metadata.get("max_tokens") != 1536
            or policy_metadata.get("task_goals") is not False or policy_metadata.get("policy_override") is not False):
        raise ValueError("Loaded policy is not the evaluated M8 representation/layout")
    return report
