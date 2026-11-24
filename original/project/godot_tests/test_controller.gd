extends SceneTree
## Controller contract tests use fabricated replies, never claimed as model evidence.

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
	main.paused = true
	main.mode_buttons.human.pressed.emit()
	check(main.mode == "human", "Human button must select human mode")
	main.mode_buttons.baseline.pressed.emit()
	check(main.mode == "baseline", "Baseline button must select baseline mode")
	main.mode_buttons.model.pressed.emit()
	check(main.mode == "model", "Model button must select model mode")
	main.source = "model"
	main._set_mode("human")
	main._set_mode("model")
	check(main.source == "none", "New model session must not reuse old success attribution")
	var tick_before: int = main.world.tick
	main.request_pending = true
	main.request_tick = tick_before
	main._on_model_reply(HTTPRequest.RESULT_SUCCESS, 200, PackedStringArray(), '{"action":{"verb":"wait"},"source":"model","raw_valid":true}'.to_utf8_buffer())
	check(main.world.tick == tick_before, "Missing tick must not advance simulation")
	main.request_pending = true
	main.request_tick = tick_before
	main._on_model_reply(HTTPRequest.RESULT_SUCCESS, 200, PackedStringArray(), JSON.stringify({"tick": tick_before, "action": {"verb": "teleport"}, "source": "model", "raw_valid": true}).to_utf8_buffer())
	check(main.world.tick == tick_before and main.source == "none" and main.paused, "Invalid model output must pause without a replacement action")
	tick_before = main.world.tick
	main.request_pending = true
	main.request_tick = tick_before
	main._on_model_reply(HTTPRequest.RESULT_SUCCESS, 200, PackedStringArray(), JSON.stringify({"tick": tick_before - 1, "action": {"verb": "wait"}, "source": "model"}).to_utf8_buffer())
	check(main.world.tick == tick_before, "Stale reply must not advance simulation")
	main.request_pending = true
	main.request_tick = tick_before
	main._on_model_reply(HTTPRequest.RESULT_TIMEOUT, 0, PackedStringArray(), PackedByteArray())
	check(main.source == "none" and main.world.tick == tick_before and main.paused, "Timeout must pause without advancing the world")
	main.request_pending = true
	main.request_tick = main.world.tick
	main._on_model_reply(HTTPRequest.RESULT_SUCCESS, 200, PackedStringArray(), JSON.stringify({"tick": main.world.tick, "action": {"verb": "rest"}, "source": "baseline", "reason": "test comparator"}).to_utf8_buffer())
	check(main.source == "none" and main.world.tick == tick_before and main.paused, "Model mode must reject a substituted baseline action")
	check(main.sample_count == 0 and main.fallback_count == 0, "Rejected responses must not become action samples")
	main.request_pending = true
	main.request_tick = main.world.tick
	tick_before = main.world.tick
	main._on_model_reply(HTTPRequest.RESULT_SUCCESS, 200, PackedStringArray(), JSON.stringify({"tick": main.world.tick, "action": {"verb": "wait"}, "source": "model", "raw_valid": true, "reason": "fabricated test fixture"}).to_utf8_buffer())
	check(main.source == "model" and main.world.tick == tick_before + 1, "Valid source-marked protocol fixture must execute exactly once")
	main.paused = false
	main.request_pending = true
	main._toggle_pause()
	check(main.paused and not main.request_pending, "Pause must cancel in-flight inference")
	main._restart()
	check(main.world.tick == 0 and not main.world.dead, "Restart must restore a new live world")
	await _test_paused_movement(main)
	main.queue_free()
	await process_frame
	print("CONTROLLER_TESTS ", "PASS" if failures == 0 else "FAIL", " failures=", failures)
	quit(0 if failures == 0 else 1)

func _test_paused_movement(main) -> void:
	main._set_mode("human")
	main._toggle_pause()
	main.manual_target = Vector2i(49, 48)
	var space := InputEventKey.new()
	space.keycode = KEY_SPACE
	space.pressed = true
	await physics_frame
	await physics_frame
	main._unhandled_input(space)
	var one_tick: int = main.world.tick
	check(one_tick == 1, "Paused Space must execute one simulation move")
	for i in range(90):
		await physics_frame
	check(main.paused and main.world.tick == one_tick, "Paused single step must never start another simulation action")
	check(not main.physics_world.is_moving(), "Paused Space must let its physical move reach the target")
	check(main.physics_world.actor_position_px().distance_to(Vector2(main.world.player) * main.world.TILE_SIZE) < 0.1, "Paused step body and logical player must agree")
	main.manual_target = Vector2i(50, 48)
	main._unhandled_input(space)
	for i in range(90):
		await physics_frame
	check(main.world.tick == one_tick + 1 and not main.physics_world.is_moving(), "Second paused Space must not remain blocked by a frozen first move")

	main._restart()
	main._set_mode("human")
	main.manual_target = Vector2i(49, 48)
	main._step()
	await physics_frame
	await physics_frame
	main._toggle_pause()
	var stopped: Vector2 = main.physics_world.player_body.position
	for i in range(10):
		await physics_frame
	check(main.physics_world.player_body.position == stopped, "Pause mid-stride must freeze actual body immediately")
	one_tick = main.world.tick
	main._unhandled_input(space)
	for i in range(90):
		await physics_frame
	check(main.world.tick == one_tick and not main.physics_world.is_moving() and main.paused, "Space mid-stride completes only the already accepted action")
	var click := InputEventMouseButton.new()
	click.button_index = MOUSE_BUTTON_LEFT
	click.pressed = true
	click.position = main.world_view.world_to_screen(Vector2(50, 48) * main.world.TILE_SIZE)
	main._unhandled_input(click)
	for i in range(90):
		await physics_frame
	check(main.world.tick == one_tick + 1 and main.world.player == Vector2i(50, 48), "Paused world click must execute one manual route step")
	check(main.paused and not main.physics_world.is_moving(), "Paused click must finish physics while remaining paused")
	one_tick = main.world.tick
	main._manual_action("rest")
	check(main.world.tick > one_tick and main.paused and not main.physics_world.is_moving(), "Paused action buttons must execute without unpausing or leaving pending movement")
