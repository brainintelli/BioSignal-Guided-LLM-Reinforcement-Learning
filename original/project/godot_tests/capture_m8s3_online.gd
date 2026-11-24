extends SceneTree
## Bounded real HTTP decisions in the actual rendered game, with no action override.
const MainScene = preload("res://scenes/main.tscn")
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
	var destination: String = ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--acceptance-output="):
			destination = arg.trim_prefix("--acceptance-output=")
	if DisplayServer.get_name() == "headless" or destination.is_empty():
		push_error("Real rendering driver and explicit acceptance output required")
		quit(1); return
	var receipt_path: String = destination.path_join("game_receipt.json")
	if FileAccess.file_exists(receipt_path):
		push_error("Refusing to overwrite prior game evidence")
		quit(1); return
	check(DirAccess.make_dir_recursive_absolute(destination) == OK, "Output directory writable")
	root.mode = Window.MODE_WINDOWED
	root.size = Vector2i(1440, 900)
	root.content_scale_size = Vector2i(1440, 900)
	var scene = MainScene.instantiate()
	root.add_child(scene)
	scene.set_process(false)
	scene.muted = true
	scene._stop_audio()
	await process_frame
	check(scene.world.profile == "open4" and scene.world.observation().world_version == 6, "Actual open4/world6")
	check(scene.allow_model_validation and scene.expected_actor_sha256.length() == 64 and scene.expected_registration_sha256.length() == 64, "Explicit actor and registration pins required")
	check(scene.world.tick == 0 and scene.sample_count == 0 and not scene.request_pending, "Journey begins before any action")
	check(scene.vital_bars.size() == 3 and not scene.vital_bars.has("energy"), "Only three vital bars")
	check(scene.session_log != null, "Durable client log required")
	scene.require_model_context = true
	scene._set_mode("model")
	var began: int = Time.get_ticks_msec()
	while failures == 0 and scene.sample_count < 32 and Time.get_ticks_msec() - began < 180000:
		if scene.paused or scene.world.dead or not scene.current_model_error.is_empty():
			check(false, "Online journey stopped: " + scene.current_model_error)
			break
		if not scene.request_pending and not scene.physics_world.is_moving():
			scene._request_model()
		await process_frame
	check(scene.sample_count == 32 and not scene.request_pending, "Exactly32 actual HTTP model actions")
	check(scene.source == "model" and scene.fallback_count == 0 and scene.model_error_count == 0, "Every action came from accepted model")
	check(scene.physics_world.unexpected_collisions == 0, "Native movement agrees with grid")
	for frame in range(24): await physics_frame
	for key in ["health", "hunger", "thirst"]:
		check(is_equal_approx(float(scene.vital_bars[key].value), float(scene.world.get(key))), "Rendered HUD matches actual " + key)
	for frame in range(3):
		await process_frame
		await RenderingServer.frame_post_draw
	var image: Image = root.get_texture().get_image()
	var image_path: String = destination.path_join("game_view.png")
	check(not image.is_empty(), "Real Viewport produced pixels")
	check(image.save_png(image_path) == OK, "Save actual rendered pixels")
	var client_path: String = destination.path_join("client.jsonl")
	if scene.session_log != null:
		scene.session_log.flush()
		var client = FileAccess.open(client_path, FileAccess.WRITE)
		check(client != null, "Save client evidence")
		if client != null:
			client.store_string(FileAccess.get_file_as_string(scene.session_log.get_path_absolute()))
			client.close()
	var obs: Dictionary = scene.world.observation()
	var receipt: Dictionary = {"passed": failures == 0, "assertions": assertions, "failures": failures,
		"source": "Actual Godot scene, HTTP responses and own Viewport texture", "synthetic_model": false,
		"agent_context": scene._model_context(), "actor_sha256": scene.expected_actor_sha256,
		"registration_sha256": scene.expected_registration_sha256, "profile": obs.profile,
		"world_version": obs.world_version, "environment_version": obs.environment_version, "balance_sha256": obs.balance_sha256,
		"decisions": scene.sample_count, "ticks": scene.world.tick, "vitals": scene._runtime_vitals(),
		"model_errors": scene.model_error_count, "fallbacks": scene.fallback_count, "unexpected_collisions": scene.physics_world.unexpected_collisions,
		"display_server": DisplayServer.get_name(), "windowed": root.mode == Window.MODE_WINDOWED,
		"width": image.get_width(), "height": image.get_height(), "image_sha256": FileAccess.get_sha256(image_path),
		"client_sha256": FileAccess.get_sha256(client_path), "client_path": client_path,
		"main_sha256": FileAccess.get_sha256("res://scripts/main.gd"),
		"driver_sha256": FileAccess.get_sha256("res://godot_tests/capture_m8s3_online.gd"),
		"scope": "32-step rendered online acceptance prefix; not independent survival performance"}
	var file = FileAccess.open(receipt_path, FileAccess.WRITE)
	if file == null: check(false, "Save render receipt")
	else:
		file.store_string(JSON.stringify(receipt, "\t")); file.close()
	scene.queue_free()
	await process_frame
	print("M8S3_ONLINE_CAPTURE ", "PASS" if failures == 0 else "FAIL", " assertions=", assertions, " failures=", failures)
	quit(0 if failures == 0 else 1)
