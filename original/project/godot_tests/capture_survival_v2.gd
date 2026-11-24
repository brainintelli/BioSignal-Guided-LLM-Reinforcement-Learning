extends SceneTree
## Export only this game's own Viewport; no desktop/window capture APIs.
const MainScene = preload("res://scenes/main.tscn")
const IMAGE_PATH: String = "res://output/rules_v2_20261108/game_view.png"
const RECEIPT_PATH: String = "res://output/rules_v2_20261108/game_view.json"
var failures: int = 0
var assertions: int = 0

func check(condition: bool, message: String) -> void:
	assertions += 1
	if not condition:
		failures += 1
		push_error(message)

func _initialize() -> void:
	call_deferred("run")

func run() -> void:
	if DisplayServer.get_name() == "headless":
		push_error("Viewport capture requires a real windowed rendering driver")
		quit(1)
		return
	root.mode = Window.MODE_WINDOWED
	root.size = Vector2i(1440, 900)
	root.content_scale_size = Vector2i(1440, 900)
	var scene = MainScene.instantiate()
	root.add_child(scene)
	await process_frame
	scene.set_process(false)
	scene.muted = true
	scene._stop_audio()
	check(root.mode == Window.MODE_WINDOWED, "Game must render in windowed mode")
	check(scene.world.profile == "open4" and scene.world.observation().world_version == 6, "Capture must use current world")
	check(scene.mode == "human" and not scene.request_pending, "Capture must not invoke a model service")
	check(scene.vital_bars.size() == 3 and not scene.vital_bars.has("energy"), "Only three reserve meters may be present")
	if failures:
		scene.queue_free()
		quit(1)
		return
	assert(scene.world.profile == "open4")
	assert(scene.vital_bars.size() == 3)
	var origin: Vector2i = scene.world.player
	var before_tick: int = scene.world.tick
	var selected: Dictionary = {}
	for action in scene.world.allowed_actions():
		if action.verb == "move":
			selected = action
			break
	check(not selected.is_empty(), "A real legal move must be available")
	if not selected.is_empty():
		scene._execute(selected, "human", "移动一格")
	var movement: Vector2i = scene.world.player - origin
	check(scene.world.tick == before_tick + 1 and absi(movement.x) + absi(movement.y) == 1, "Rendered game must execute exactly one tile and one tick")
	for key in ["health", "thirst", "hunger"]:
		check(is_equal_approx(float(scene.vital_bars[key].value), float(scene.world.get(key))), "HUD meter matches world reserve: " + key)
	# Let animation, font uploads and layout settle in the actual rendering path.
	for frame in range(24):
		await physics_frame
	for frame in range(3):
		await process_frame
		await RenderingServer.frame_post_draw
	var image: Image = root.get_texture().get_image()
	check(not image.is_empty() and image.get_width() > 0 and image.get_height() > 0, "Viewport must return rendered pixels")
	var destination: String = ProjectSettings.globalize_path("res://output/rules_v2_20261108")
	check(DirAccess.make_dir_recursive_absolute(destination) == OK, "Artifact directory must be writable")
	if failures == 0:
		check(image.save_png(IMAGE_PATH) == OK, "Rendered image must be saved")
	var receipt: Dictionary = {"source":"Godot own Viewport texture", "profile":scene.world.profile,
		"world_version":scene.world.observation().world_version, "display_server":DisplayServer.get_name(),
		"windowed":root.mode == Window.MODE_WINDOWED, "image":ProjectSettings.globalize_path(IMAGE_PATH),
		"width":image.get_width(), "height":image.get_height(), "action":selected,
		"tick_before":before_tick, "tick_after":scene.world.tick, "vitals":scene._runtime_vitals(),
		"assertions":assertions, "failures":failures, "formal_training_started":false}
	var file = FileAccess.open(RECEIPT_PATH, FileAccess.WRITE)
	check(file != null, "Render receipt must be writable")
	receipt["assertions"] = assertions
	receipt["failures"] = failures
	if file != null:
		file.store_string(JSON.stringify(receipt, "\t"))
		file.close()
	scene.queue_free()
	await process_frame
	print("SURVIVAL_V2_VIEWPORT_CAPTURE ", "PASS" if failures == 0 else "FAIL", " assertions=", assertions, " failures=", failures)
	quit(0 if failures == 0 else 1)
