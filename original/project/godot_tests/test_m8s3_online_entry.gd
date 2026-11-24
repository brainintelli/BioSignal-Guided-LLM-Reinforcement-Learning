extends SceneTree
## Synthetic refusal cases exercise the real client; no model-performance claim.
var failures: int = 0
var assertions: int = 0

func check(value: bool, message: String) -> void:
	assertions += 1
	if not value:
		failures += 1
		push_error(message)

func _initialize() -> void:
	call_deferred("run")

func run() -> void:
	var scene = load("res://scenes/main.tscn").instantiate()
	root.add_child(scene)
	await process_frame
	scene.set_process(false)
	scene.muted = true
	scene._stop_audio()
	check(scene.world.profile == "open4", "Online acceptance must use open4")
	check(scene.has_method("_model_health_authorized"), "Client needs explicit bounded-validation gate")
	if not scene.has_method("_model_health_authorized"):
		scene.queue_free(); quit(1); return
	var obs: Dictionary = scene.world.observation()
	var identity: Dictionary = {"profile": obs.profile, "world_version": obs.world_version,
		"environment_version": obs.environment_version, "balance_sha256": obs.balance_sha256,
		"actor_sha256": "a".repeat(64), "registration_sha256": "b".repeat(64)}
	var health: Dictionary = identity.duplicate()
	health.merge({"release_authorized": false, "validation_authorized": true,
		"numerical_gate_passed": true, "decision_ready": true})
	check(not scene._model_health_authorized(health), "Ordinary game rejects validation-only service")
	scene.allow_model_validation = true
	check(not scene._model_health_authorized(health), "Flag without actor and registration pins cannot authorize")
	scene.expected_actor_sha256 = identity.actor_sha256
	scene.expected_registration_sha256 = identity.registration_sha256
	check(scene._model_health_authorized(health), "Explicit validation pins accept exact offline-approved service")
	for change in [{"actor_sha256": "c".repeat(64)}, {"registration_sha256": "c".repeat(64)},
		{"profile": "open3"}, {"numerical_gate_passed": false}, {"validation_authorized": false},
		{"decision_ready": false}]:
		var bad: Dictionary = health.duplicate()
		bad.merge(change, true)
		check(not scene._model_health_authorized(bad), "Validation fails closed: " + str(change))
	var published: Dictionary = health.duplicate()
	published.release_authorized = true
	published.actor_sha256 = "c".repeat(64)
	check(not scene._model_health_authorized(published), "Release flag cannot bypass explicit validation pins")
	scene.allow_model_validation = false
	check(scene._model_health_authorized(published), "Ordinary published service retains existing gate")
	scene.allow_model_validation = true
	var reply: Dictionary = identity.duplicate()
	reply.merge({"broker_version": "m8s3.0", "tick": 0, "action": {"verb": "wait"},
		"source": "model", "raw_valid": true})
	for mutation in [{}, {"agent_context": scene._model_context(), "actor_sha256": "c".repeat(64)},
		{"agent_context": scene._model_context(), "registration_sha256": "c".repeat(64)}]:
		var bad: Dictionary = reply.duplicate()
		bad.merge(mutation, true)
		scene.mode = "model"; scene.request_pending = true; scene.request_tick = 0
		scene.balanced_health_pending = false; scene.require_model_context = false; scene.paused = false
		scene._on_model_reply(HTTPRequest.RESULT_SUCCESS, 200, PackedStringArray(), JSON.stringify(bad).to_utf8_buffer())
		check(scene.world.tick == 0 and scene.paused and scene.source == "none", "Bad decision cannot advance actual world")
	scene.queue_free()
	await process_frame
	print("M8S3_ONLINE_ENTRY ", "PASS" if failures == 0 else "FAIL", " assertions=", assertions, " failures=", failures)
	quit(0 if failures == 0 else 1)
