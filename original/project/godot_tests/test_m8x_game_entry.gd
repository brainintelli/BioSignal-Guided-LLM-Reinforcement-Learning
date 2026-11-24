extends SceneTree
const Exploration = preload("res://scripts/exploration_world.gd")
var failures: int = 0
func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)
func _initialize() -> void:
	call_deferred("run")
func run() -> void:
	# Exercise the real reset gate, without modifying shared on-disk config.
	for patch in [{"rest_energy":28},{"resource_regrowth_ticks":120},{"food_restore":99},{"thirst_drain":0.4},{"hunger_drain":NAN},{"night_damage":-1},{"world_version":2},{"profile":"open1"}]:
		var bad=Exploration.new()
		bad.balance.merge(patch,true)
		bad.reset(2304,"open3")
		check(not bad.rules_valid and bad.dead and bad.allowed_actions().is_empty(), "Invalid config fails closed: "+str(patch))
		check(not bad.apply_action({"verb":"wait"}), "Invalid world cannot act")
	var scene=load("res://scenes/main.tscn").instantiate()
	root.add_child(scene)
	await process_frame
	scene.set_process(false)
	# Keep test logging within the workspace when user:// is sandbox-restricted.
	if scene.session_log == null:
		scene.session_log = FileAccess.open("res://output/runtime_tests/test_m8x_game_entry_session_%s.jsonl" % Time.get_ticks_usec(), FileAccess.WRITE)
	if scene.session_log == null:
		check(false, "Test session log could not open"); scene.queue_free(); quit(1); return
	check(scene.world is Exploration and scene.world.rules_valid and scene.mode=="human", "Explicit open3 enters manual balanced world")
	if not scene.world is Exploration:
		scene.queue_free(); quit(1); return
	check(scene.world.observation().world_version==5 and scene.world.observation().profile=="open3", "Version/profile identity")
	var loaded_hash: String=scene.world.balance_sha256
	check(loaded_hash==FileAccess.get_sha256("res://configs/m8_exploration_v1.json"), "Hash is captured from loaded config bytes")
	check(scene.difficulty_button.text.contains("试验规则") and scene.mode_label.text.contains("探索生存 · 试验规则"), "Experimental UI label is visible even with observer closed")
	check(scene.progress_label.text.contains("水耗 0.08") and scene.progress_label.text.contains("饥饿消耗 0.20"), "HUD shows the actual exploration consumption rates")
	scene._restart()
	check(scene.world is Exploration and scene.world.tick==0, "Restart retains balanced subclass")
	scene._toggle_difficulty()
	check(not scene.world is Exploration and scene.world.profile=="d0", "Easy switch uses original subclass")
	scene._toggle_difficulty()
	check(scene.world is Exploration and scene.world.profile=="open3", "Switch back restores balanced subclass")
	for boundary in [900,1800,5400]:
		scene.world.tick=boundary-1
		scene.world.health=100;scene.world.hunger=100;scene.world.thirst=100;scene.world.energy=100
		scene.paused=false
		scene._execute({"verb":"wait"},"human","synthetic boundary test; not model performance")
		check(scene.world.tick==boundary and not scene.world.dead and not scene._lesson_finished(), "No tick/day cap at "+str(boundary))
		check(scene._runtime_status().runtime_tick_limit==null and scene._runtime_status().runtime_day_limit==null, "Unlimited runtime declaration")
	scene.session_log.flush()
	var path: String=scene.session_log.get_path_absolute()
	var lines: PackedStringArray=FileAccess.get_file_as_string(path).strip_edges().split("\n")
	var row: Dictionary=JSON.parse_string(lines[-1])
	check(row.world_version==5 and row.environment_version=="m8_exploration_v1" and row.balance_sha256==loaded_hash, "Client log identifies actual in-memory rules")
	check(not scene._balanced_identity_matches({"profile":"open1","world_version":2}), "Old service rejected")
	check(scene._balanced_identity_matches({"profile":"open3","world_version":5,"environment_version":"m8_exploration_v1","balance_sha256":loaded_hash}), "Exact new identity matches")
	scene.mode="model";scene.request_pending=true;scene.balanced_health_pending=true;scene.request_tick=scene.world.tick
	scene._on_model_reply(HTTPRequest.RESULT_SUCCESS,200,PackedStringArray(),JSON.stringify({"environment_profile":"open1","environment_version":2,"release_authorized":true,"decision_ready":true}).to_utf8_buffer())
	check(not scene.request_pending and scene.paused and scene.source=="none", "Mismatched health stops before model decision POST")
	check(scene.world.tick==5400, "Health refusal cannot advance world")
	# The broker identity itself must require context, including direct scene entry.
	scene._set_mode("human");scene._restart();scene.set_process(false)
	scene.mode="model";scene.request_pending=true;scene.balanced_health_pending=false
	scene.request_tick=0;scene.require_model_context=false
	var no_context: Dictionary={"profile":"open3","world_version":5,"environment_version":"m8_exploration_v1","balance_sha256":loaded_hash,"broker_version":"m8x.0","tick":0,"action":{"verb":"wait"},"source":"model","raw_valid":true}
	scene._on_model_reply(HTTPRequest.RESULT_SUCCESS,200,PackedStringArray(),JSON.stringify(no_context).to_utf8_buffer())
	check(scene.world.tick==0 and scene.paused and scene.source=="none", "Exploration broker without journey context cannot advance world")
	scene._set_mode("human");scene.paused=false;scene._restart();scene.set_process(false)
	if DisplayServer.get_name()!="headless":
		await RenderingServer.frame_post_draw
		root.get_texture().get_image().save_png("res://output/runtime_tests/game_entry.png")
	scene.muted=true;scene._stop_audio()
	await create_timer(0.06).timeout
	scene.queue_free()
	await process_frame
	print("M8X_GAME_ENTRY ","PASS" if failures==0 else "FAIL"," failures=",failures," synthetic_boundaries_not_model_results=true")
	quit(0 if failures==0 else 1)
