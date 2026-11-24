extends SceneTree
## Fabricated transport tests, never evidence of language-model survival.
const MainScene = preload("res://scenes/main.tscn")
var failures: int = 0

func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var main = MainScene.instantiate()
	root.add_child(main)
	main.set_process(false)
	if not main.has_method("_model_context"):
		push_error("Per-journey model transport identity is missing")
		quit(1)
		return
	var first: Dictionary = main._model_context()
	check(not str(first.session_id).is_empty(), "Client session must have an explicit identity")
	check(first == main._model_context(), "Observation retries must retain the same journey identity")
	check(not main.world.observation().has("agent_context"), "Transport identity must stay outside actor observations")
	main._set_mode("model")
	var second: Dictionary = main._model_context()
	check(first.session_id == second.session_id and second.episode > first.episode, "Controller handoff starts a new sampling stream")
	main._restart()
	var third: Dictionary = main._model_context()
	check(third.episode > second.episode, "Restart starts a new episode stream")
	main.paused = false
	main.physics_world.frozen = false
	var tick_before: int = main.world.tick
	var samples_before: int = main.sample_count
	main.request_pending = true
	main.request_tick = tick_before
	main._on_model_reply(HTTPRequest.RESULT_TIMEOUT, 0, PackedStringArray(), PackedByteArray())
	check(main.paused and main.world.tick == tick_before, "Model timeout pauses without executing a world action")
	check(main.sample_count == samples_before and main.fallback_count == 0, "Transport errors must not become model or fallback samples")
	check(main.model_error_count == 1 and main.source == "none", "Transport failure is recorded separately")
	main.require_model_context = true
	main.request_pending = true
	main.request_tick = tick_before
	main._on_model_reply(HTTPRequest.RESULT_SUCCESS, 200, PackedStringArray(), JSON.stringify({"tick": tick_before, "action": {"verb": "wait"}, "source": "model", "raw_valid": true}).to_utf8_buffer())
	check(main.world.tick == tick_before and main.model_error_count == 2, "A context-required connection rejects a legacy reply with no journey identity")
	main.request_pending = true
	main.request_tick = tick_before
	main._on_model_reply(HTTPRequest.RESULT_SUCCESS, 200, PackedStringArray(), JSON.stringify({"tick": tick_before, "agent_context": second, "action": {"verb": "wait"}, "source": "model", "raw_valid": true}).to_utf8_buffer())
	check(main.world.tick == tick_before, "A previous journey reply must not execute even when tick matches")
	main.require_model_context = false
	for version in ["m6.0", "m7.0", "m8.0"]:
		main.request_pending = true
		main.request_tick = tick_before
		var errors_before: int = main.model_error_count
		main._on_model_reply(HTTPRequest.RESULT_SUCCESS, 200, PackedStringArray(), JSON.stringify({"tick": tick_before, "broker_version": version, "action": {"verb": "wait"}, "source": "model", "raw_valid": true}).to_utf8_buffer())
		check(main.world.tick == tick_before and main.model_error_count == errors_before + 1, version + " must require journey context even without a launcher flag")
	main.request_pending = true
	main.request_tick = tick_before
	main._on_model_reply(HTTPRequest.RESULT_SUCCESS, 200, PackedStringArray(), JSON.stringify({"tick": tick_before, "agent_context": third, "action": {"verb": "wait"}, "source": "model", "raw_valid": true, "rng_seed": "9223372036854775806"}).to_utf8_buffer())
	check(main.world.tick == tick_before + 1 and main.source == "model", "Matching source-marked fixture executes exactly once")
	check(main.last_model_decision.get("rng_seed", "") == "9223372036854775806", "Live RNG identity must retain all decimal digits through Godot JSON")
	main.queue_free()
	await process_frame
	print("MODEL_SESSION_TESTS ", "PASS" if failures == 0 else "FAIL", " failures=", failures)
	quit(0 if failures == 0 else 1)
