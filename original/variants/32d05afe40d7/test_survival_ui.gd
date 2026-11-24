extends SceneTree
## UI behavior and layout checks; fabricated protocol replies are not model evidence.
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
	await process_frame
	check(main.lesson_id == "open" and main.world.profile == "open1", "Default must be task-free open1 with standard survival rules")
	check(main.mode == "human" and main.world.tick == 0, "Default must not run hidden rule actions")
	check(not main.lesson_select.is_visible_in_tree() and not main.lesson_next.is_visible_in_tree(), "Teaching navigation must not appear in survival HUD")
	check(not main.sidebar.visible, "Observer should start collapsed to expose the world")
	main.observer_button.pressed.emit()
	check(main.sidebar.visible, "Observer toggle opens panel")
	main.observer_button.pressed.emit()
	check(not main.sidebar.visible, "Observer toggle closes panel")
	for size in [Vector2i(1280, 720), Vector2i(1440, 900), Vector2i(1920, 1080)]:
		root.size = size
		root.content_scale_size = size
		main._layout()
		await process_frame
		await process_frame
		main._layout()
		var viewport: Rect2 = main.get_viewport_rect()
		check(main.world_view.view_size == viewport.size, "World must use full viewport at " + str(size))
		if not main.observer_open:
			main._toggle_observer()
		await process_frame
		await process_frame
		for panel in [main.hud_top, main.hud_bottom, main.sidebar, main.inventory_panel]:
			check(viewport.grow(1).encloses(panel.get_global_rect()), "Panel outside viewport at " + str(size) + ": " + panel.name + " " + str(panel.get_global_rect()))
		check(not main.hud_top.get_global_rect().intersects(main.hud_bottom.get_global_rect()), "Header and dock overlap at " + str(size))
		check(not main.sidebar.get_global_rect().intersects(main.hud_bottom.get_global_rect()), "Observer covers controls at " + str(size))
		main._toggle_observer()
	main.mode_buttons.model.pressed.emit()
	check(main.mode == "model" and main.sidebar.visible, "Model mode exposes its attribution panel")
	main.request_pending = true
	main.request_tick = main.world.tick
	main._on_model_reply(HTTPRequest.RESULT_SUCCESS, 200, PackedStringArray(), JSON.stringify({"tick": main.world.tick, "action": {"verb": "wait"}, "source": "model", "raw_valid": true, "model": "fixture-large-model", "latency_ms": 123, "reason": "fabricated UI fixture"}).to_utf8_buffer())
	check(main.connected_model == "fixture-large-model" and main.source_label.text.contains("fixture-large-model"), "Actual reply model identity must be visible")
	check(main.latency_label.text.contains("123 ms") and main.mode_label.text.contains("真实模型"), "Model latency and source must be shown")
	main.mode_buttons.human.pressed.emit()
	check(main.connected_model.is_empty() and not main.source_label.text.contains("fixture-large-model"), "Switching controllers clears stale model identity")
	main._restart()
	main._toggle_pause()
	main.mode_buttons.human.grab_focus()
	main.manual_target = Vector2i(49, 48)
	await physics_frame
	await physics_frame
	var space := InputEventKey.new()
	space.keycode = KEY_SPACE
	space.pressed = true
	root.push_input(space)
	for frame in range(90):
		await physics_frame
	check(main.paused and main.mode == "human" and main.world.tick == 1 and main.world.player == Vector2i(49, 48), "Focused HUD must preserve exactly one paused world step")
	main.queue_free()
	await process_frame
	print("SURVIVAL_UI_TESTS ", "PASS" if failures == 0 else "FAIL", " failures=", failures)
	quit(0 if failures == 0 else 1)
