extends Node2D

const Curriculum = preload("res://scripts/curriculum.gd")
const World = preload("res://scripts/world_state.gd")
const ChallengeWorld = preload("res://scripts/challenge_world.gd")
const BalancedWorld = preload("res://scripts/balanced_world.gd")
const ExplorationWorld = preload("res://scripts/exploration_world.gd")
const SurvivalWorld = preload("res://scripts/survival_world.gd")
var balanced_health_pending: bool = false
const PhysicsWorld = preload("res://scripts/physics_world.gd")
const WorldView = preload("res://scripts/world_view.gd")
const INK := Color("2b4d42")
const MUTED := Color("7b8e7e")
const PAPER := Color("f9fbf0")
const ACCENT := Color("3b8868")

var curriculum = Curriculum.new()
var lesson_id: String = "open"
var open_difficulty: String = "open4"
var standard_difficulty: String = "open4"
var lesson_select: OptionButton
var lesson_label: Label
var lesson_ledger: Label
var lesson_initial: Label
var lesson_progress: Label
var lesson_next: Button
var world: MeadowWorld = World.new()
var world_view: Node2D
var physics_world: Node2D
var difficulty_button: Button
var collision_label: Label
var held_direction: Vector2i = Vector2i.ZERO
var ui_root: Control
var sidebar: PanelContainer
var title_label: Label
var phase_label: Label
var mode_label: Label
var source_label: Label
var event_label: Label
var inventory_label: Label
var progress_label: Label
var camp_label: Label
var log_label: Label
var latency_label: Label
var pause_button: Button
var speed_button: Button
var sound_button: Button
var mode_buttons: Dictionary = {}
var vital_bars: Dictionary = {}
var vital_labels: Dictionary = {}
var context_buttons: Array[Button] = []
var day_bar: ProgressBar
var http: HTTPRequest
var music: AudioStreamPlayer
var effects: AudioStreamPlayer
var mode: String = "human"
var source: String = "human"
var source_detail: String = "WASD 移动，点击资源采集"
var paused: bool = false
var muted: bool = false
var speed: int = 1
var timer: float = 0.0
var request_pending: bool = false
var request_tick: int = -1
var request_epoch: int = 0
var control_session_id: String = "%s-%s-%s" % [OS.get_process_id(), Time.get_ticks_usec(), int(Time.get_unix_time_from_system() * 1000)]
var model_error_count: int = 0
var current_model_error: String = ""
var require_model_context: bool = false
var allow_model_validation: bool = false
var expected_actor_sha256: String = ""
var expected_registration_sha256: String = ""
var manual_target: Vector2i = Vector2i(-1, -1)
var manual_resource: String = ""
var logs: Array[String] = []
var elapsed: float = 0.0
var screenshot_requested: bool = false
var smoke_finishing: bool = false
var smoke_mode: bool = false
var smoke_duration: float = 10.0
var last_action: String = "观察"
var last_latency: float = 0.0
var sample_count: int = 0
var fallback_count: int = 0
var session_log: FileAccess
var music_is_night: bool = false
var broker_url: String = "http://127.0.0.1:8765"
var observer_button: Button
var observer_open: bool = false
var connected_model: String = ""
var last_model_decision: Dictionary = {}
var hud_frame: MarginContainer
var hud_top: PanelContainer
var hud_bottom: PanelContainer
var inventory_panel: PanelContainer
var legacy_course_ui: Control

func _ready() -> void:
	var difficulty: String = "open4"
	var seed_value: int = 2304
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--difficulty="):
			difficulty = arg.trim_prefix("--difficulty=")
		elif arg.begins_with("--seed="):
			seed_value = int(arg.trim_prefix("--seed="))
		elif arg == "--validate-model":
			allow_model_validation = true
		elif arg.begins_with("--expected-actor="):
			expected_actor_sha256 = arg.trim_prefix("--expected-actor=")
		elif arg.begins_with("--expected-registration="):
			expected_registration_sha256 = arg.trim_prefix("--expected-registration=")
	open_difficulty = difficulty
	if difficulty in ["m7_control", "m7_terrain", "open2", "open3", "open4"]:
		standard_difficulty = difficulty
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--lesson="):
			lesson_id = arg.trim_prefix("--lesson=")
	_reset_open_world(seed_value, difficulty)
	if (world is BalancedWorld or world is ExplorationWorld or world is SurvivalWorld) and not world.rules_valid:
		get_tree().quit(1)
		return
	if difficulty in ["open2", "open3", "open4"]: lesson_id = "open"
	if lesson_id != "open" and not curriculum.configure(world, lesson_id, seed_value):
		lesson_id = "open"
	physics_world = PhysicsWorld.new()
	physics_world.world = world
	add_child(physics_world)
	world_view = WorldView.new()
	world_view.world = world
	world_view.physics_world = physics_world
	add_child(world_view)
	world_view.snap_camera()
	http = HTTPRequest.new()
	http.timeout = 65.0
	add_child(http)
	http.request_completed.connect(_on_model_reply)
	_build_ui()
	_setup_audio()
	get_viewport().size_changed.connect(_layout)
	_layout()
	for arg in OS.get_cmdline_user_args():
		if arg == "--debug-collision":
			world_view.debug_collision = true
		if arg == "--smoke-test":
			smoke_mode = true
		if arg == "--screenshot":
			screenshot_requested = true
		if arg == "--model":
			_set_mode("model")
		if arg == "--require-model-context":
			require_model_context = true
		if arg == "--observer" and not observer_open:
			_toggle_observer()
		if arg == "--baseline":
			_set_mode("baseline")
		if arg.begins_with("--broker-url="):
			broker_url = arg.trim_prefix("--broker-url=").trim_suffix("/")
		if arg.begins_with("--model-timeout="):
			http.timeout = clampf(float(arg.trim_prefix("--model-timeout=")), 1.0, 120.0)
		if arg.begins_with("--smoke-duration="):
			smoke_duration = clampf(float(arg.trim_prefix("--smoke-duration=")), 1.0, 300.0)
	DirAccess.make_dir_recursive_absolute("user://sessions")
	session_log = FileAccess.open("user://sessions/session_%s.jsonl" % control_session_id, FileAccess.WRITE)
	if smoke_mode and session_log != null:
		print("SMOKE_SESSION_LOG ", session_log.get_path_absolute())
	_append_log("第 1 天 / " + ("等待模型决策" if mode == "model" else "规则演示开始" if mode == "baseline" else "新的野外生活，从这里开始"))
	_refresh_ui()

func _setup_audio() -> void:
	music = AudioStreamPlayer.new()
	music.volume_db = -16.0
	add_child(music)
	if DisplayServer.get_name() != "headless" and ResourceLoader.exists("res://assets/audio/meadow_loop.wav"):
		music.stream = load("res://assets/audio/meadow_loop.wav")
		music.finished.connect(func() -> void:
			if not muted:
				music.play())
		music.play()
	effects = AudioStreamPlayer.new()
	effects.volume_db = -13.0
	add_child(effects)

func _play_effect(verb: String) -> void:
	if muted or DisplayServer.get_name() == "headless":
		return
	var filename: String = "gather" if verb == "gather" else verb if verb in ["eat", "drink"] else "build" if verb == "build_camp" else ""
	if filename.is_empty():
		return
	var path: String = "res://assets/audio/%s.wav" % filename
	if ResourceLoader.exists(path):
		effects.stream = load(path)
		effects.play()

func _stop_audio() -> void:
	if music != null:
		music.stop()
		music.stream = null
	if effects != null:
		effects.stop()
		effects.stream = null

func _exit_tree() -> void:
	_stop_audio()
	if session_log != null:
		session_log.close()

func _process(delta: float) -> void:
	if smoke_finishing:
		return
	elapsed += delta
	if smoke_mode and elapsed > smoke_duration:
		# Finish once: frame-post-draw can suspend while another process tick occurs.
		smoke_finishing = true
		set_process(false)
		print("SMOKE_PASS tick=", world.tick, " day=", world.day(), " source=", source, " alive=", not world.dead)
		if screenshot_requested and DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			var capture: Image = get_viewport().get_texture().get_image()
			capture.save_png("res://godot_tests/smoke_capture.png")
			capture = null
		# WAV playback is released on the audio mixer thread. Stop before tree teardown
		# and allow its pending mix cycle to relinquish meadow_loop.wav references.
		muted = true
		_stop_audio()
		await get_tree().create_timer(0.05).timeout
		await get_tree().process_frame
		get_tree().quit()
		return
	if paused or world.dead or request_pending or physics_world.is_moving():
		return
	timer += delta * float(speed)
	if timer < (0.12 if mode == "human" and held_direction != Vector2i.ZERO else 0.7):
		return
	timer = 0.0
	_step()

func _step() -> void:
	if world.dead or request_pending or _lesson_finished():
		return
	if physics_world.is_moving():
		# Space while paused mid-stride completes the already accepted step only.
		if paused:
			physics_world.complete_paused_step = true
		return
	if mode == "model":
		_request_model()
	elif mode == "baseline":
		_execute(curriculum.baseline_action(world) if lesson_id != "open" else world.baseline_action(), "baseline", "公开规则课程比较器" if lesson_id != "open" else "公开规则基线选择动作")
	elif held_direction != Vector2i.ZERO:
		var action: Dictionary = {"verb": "move", "target_id": world.tile_id(world.player + held_direction)}
		if world.allowed_actions().has(action):
			_execute(action, "human", "方向键移动")
	elif manual_target.x >= 0:
		if not manual_resource.is_empty():
			var r: Dictionary = world.resource_by_id(manual_resource)
			if not r.is_empty() and world.can_interact(Vector2i(int(r.x), int(r.y))):
				_execute({"verb": "gather", "target_id": manual_resource}, "human", "点击采集")
				manual_target = Vector2i(-1, -1)
				manual_resource = ""
				return
		if world.player == manual_target:
			manual_target = Vector2i(-1, -1)
			return
		var action: Dictionary = world.step_toward(manual_target)
		if str(action.verb) == "wait":
			manual_target = Vector2i(-1, -1)
			source_detail = "目标不可达；请点击已探索陆地。"
			_refresh_ui()
		else:
			_execute(action, "human", "点击移动 · 执行寻路一步")

func _model_context() -> Dictionary:
	return {"session_id": control_session_id, "episode": request_epoch, "world_seed": world.world_seed}

func _matches_model_context(value: Variant) -> bool:
	if not value is Dictionary or value.size() != 3:
		return false
	if value.get("session_id") != control_session_id:
		return false
	for field in ["episode", "world_seed"]:
		if not value.get(field) is int and not value.get(field) is float:
			return false
	return float(value.episode) == float(request_epoch) and float(value.world_seed) == float(world.world_seed)

func _model_failed(reason: String) -> void:
	balanced_health_pending = false
	request_pending = false
	paused = true
	physics_world.frozen = true
	physics_world.complete_paused_step = false
	source = "none"
	last_latency = 0.0
	model_error_count += 1
	current_model_error = reason
	source_detail = reason + "；已暂停，可恢复后重试。"
	_append_log("%03d  模型连接异常，世界已暂停" % world.tick)
	if session_log != null:
		session_log.store_line(JSON.stringify(_identity_record({"event": "model_error", "tick": world.tick, "reason": reason, "agent_context": _model_context(), "source": "none", "runtime_status": _runtime_status(), "vitals": _runtime_vitals()})))
		session_log.flush()
	_refresh_ui()

func _request_model() -> void:
	request_pending = true
	request_tick = world.tick
	source_detail = "模型正在观察环境 · 等待期间时间暂停"
	_refresh_ui()
	if world.profile in ["open2", "open3", "open4"]:
		balanced_health_pending = true
		var health_error: Error = http.request(broker_url + "/health")
		if health_error != OK: _model_failed("试验规则模型健康检查失败")
		return
	_send_model_decision()

func _balanced_identity_matches(value: Dictionary) -> bool:
	var obs: Dictionary = world.observation()
	return world.profile in ["open2", "open3", "open4"] and str(value.get("profile", value.get("environment_profile", ""))) == obs.get("profile") and value.get("world_version") == obs.get("world_version") and value.get("environment_version") == obs.get("environment_version") and value.get("balance_sha256") == obs.get("balance_sha256")

func _validation_identity_matches(value: Dictionary) -> bool:
	return world.profile == "open4" and _balanced_identity_matches(value) and expected_actor_sha256.length() == 64 and expected_actor_sha256.is_valid_hex_number() and expected_registration_sha256.length() == 64 and expected_registration_sha256.is_valid_hex_number() and value.get("actor_sha256") == expected_actor_sha256 and value.get("registration_sha256") == expected_registration_sha256

func _model_health_authorized(health: Dictionary) -> bool:
	if not _balanced_identity_matches(health) or health.get("decision_ready") != true:
		return false
	if allow_model_validation:
		return _validation_identity_matches(health) and health.get("validation_authorized") == true and health.get("numerical_gate_passed") == true
	return health.get("release_authorized") == true

func _send_model_decision() -> void:
	var context: Dictionary = _model_context()
	var headers: PackedStringArray = ["Content-Type: application/json", "X-Agent-Session: " + str(context.session_id), "X-Agent-Episode: " + str(context.episode), "X-World-Seed: " + str(context.world_seed)]
	var error: Error = http.request(broker_url + "/decide", headers, HTTPClient.METHOD_POST, JSON.stringify(curriculum.observation(world) if lesson_id != "open" else world.observation()))
	if error != OK:
		_model_failed("模型服务未连接")

func _identity_record(row: Dictionary) -> Dictionary:
	var obs: Dictionary = world.observation()
	row["world_version"] = obs.world_version
	row["environment_version"] = obs.get("environment_version", "")
	row["balance_sha256"] = obs.get("balance_sha256", "")
	return row

func _runtime_vitals() -> Dictionary:
	if world.profile == "open4":
		return {"health": world.health, "hunger": world.hunger, "thirst": world.thirst}
	return {"health": world.health, "hunger": world.hunger, "thirst": world.thirst, "energy": world.energy}

func _runtime_status() -> Dictionary:
	# Observation only: no damage, supplies, revival, or policy override here.
	var conditions: Array[String] = []
	if world.dead and not world.last_step_trace.is_empty():
		var final_tick: Dictionary = world.last_step_trace[-1]
		if int(final_tick.tick) == world.tick and float(final_tick.damage) > 0.0:
			if float(final_tick.thirst) <= 0.0: conditions.append("dehydration")
			if float(final_tick.hunger) <= 0.0: conditions.append("starvation")
			if world.profile != "open4" and float(final_tick.get("energy", 100.0)) <= 0.0: conditions.append("exhaustion")
			if world.profile not in ["d0", "open4"] and world.is_night() and not world.camp_near():
				conditions.append("night_exposure")
			if not world.camp_near():
				for threat in world.visible_threats():
					if float(threat.distance) < 1.6:
						conditions.append("nearby_threat")
						break
	var state: String = "dead" if world.dead else "model_error" if paused and not current_model_error.is_empty() else "paused" if paused else "running"
	var reason: String = "health_depleted" if world.dead else "model_error" if state == "model_error" else "lesson_finished" if paused and _lesson_finished() else "user_pause" if paused else ""
	var names: Dictionary = {"dehydration": "缺水", "starvation": "饥饿", "exhaustion": "体力耗尽", "night_exposure": "营外夜间伤害", "nearby_threat": "附近威胁伤害"}
	var labels: Array[String] = []
	for condition in conditions: labels.append(str(names[condition]))
	return {"state": state, "stop_reason": reason, "damage_conditions": conditions,
		"death_detail": "、".join(labels) + " · 生命耗尽" if not labels.is_empty() else "生命耗尽（缺少末步伤害记录）" if world.dead else "",
		"model_error": current_model_error if state == "model_error" else "",
		"runtime_day_limit": null, "runtime_tick_limit": null}

func _on_model_reply(result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	if balanced_health_pending:
		balanced_health_pending = false
		if not request_pending or mode != "model" or world.profile not in ["open2", "open3", "open4"] or request_tick != world.tick:
			return
		var health: Variant = JSON.parse_string(body.get_string_from_utf8())
		if result != HTTPRequest.RESULT_SUCCESS or response_code != 200 or not health is Dictionary or not _model_health_authorized(health):
			_model_failed("探索生存试验规则尚无匹配的已发布模型；可继续手动生存")
			return
		_send_model_decision()
		return
	if not request_pending:
		return
	request_pending = false
	if mode != "model" or request_tick != world.tick:
		return
	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		_model_failed("连接失败 / 超时，未获得模型决策")
		return
	var decoded: Variant = JSON.parse_string(body.get_string_from_utf8())
	if not decoded is Dictionary:
		_model_failed("响应不是 JSON 对象")
		return
	var reply: Dictionary = decoded
	if world.profile in ["open2", "open3", "open4"] and not _balanced_identity_matches(reply):
		_model_failed("模型响应与当前试验规则身份不匹配")
		return
	if allow_model_validation and not _validation_identity_matches(reply):
		_model_failed("在线验收响应与指定模型或训练登记不匹配")
		return
	if (require_model_context or str(reply.get("broker_version", "")) in ["m6.0", "m7.0", "m8.0", "m8b.0", "m8x.0", "m8s3.0"]) and not reply.has("agent_context"):
		_model_failed("模型响应缺少旅程身份")
		return
	if not reply.has("tick") or not reply.tick is float and not reply.tick is int or float(reply.tick) != float(request_tick):
		_model_failed("丢弃缺少时间戳或过期响应")
		return
	if reply.has("agent_context") and not _matches_model_context(reply.agent_context):
		_model_failed("丢弃其他旅程的响应")
		return
	var action: Variant = reply.get("action", {})
	var reported_source: String = str(reply.get("source", "fallback"))
	if reported_source != "model":
		_model_failed("服务返回的动作不是模型决策")
		return
	if reply.get("raw_valid", false) != true:
		_model_failed("服务未确认原始模型输出合法")
		return
	if not action is Dictionary or not world.allowed_actions().has(action):
		_model_failed("动作不在合法列表中")
		return
	last_latency = float(reply.get("latency_ms", 0.0))
	current_model_error = ""
	connected_model = str(reply.get("model", "模型服务未返回标识")) if reported_source == "model" else ""
	last_model_decision = {}
	for field in ["model", "actor_sha256", "registration_sha256", "decoding", "agent_context", "rng_seed", "rng_before_sha256", "rng_after_sha256", "raw_output"]:
		if reply.has(field):
			last_model_decision[field] = reply[field]
	_execute(action, reported_source, str(reply.get("reason", "代理返回动作")).left(160))

func _execute(action: Dictionary, actor_source: String, reason: String) -> void:
	if action.is_empty() or physics_world.is_moving() or _lesson_finished():
		return
	if str(action.get("verb", "")) in ["move", "navigate"]:
		var destination: Vector2i = world.tile_from_id(str(action.get("target_id", "")))
		if str(action.verb) == "navigate":
			for landmark in world.navigation_landmarks():
				if str(landmark.id) == str(action.target_id):
					var step: Dictionary = world.step_toward(Vector2i(int(landmark.x), int(landmark.y)))
					destination = world.tile_from_id(str(step.get("target_id", "")))
		if not world.is_walkable(destination) or not physics_world.can_move_to(destination):
			if actor_source == "model":
				_model_failed("模型移动与实际碰撞状态不一致")
				return
			source_detail = "前方有实体阻挡；移动未执行。"
			_refresh_ui()
			return
	var before: int = world.tick
	if not world.apply_action(action):
		if actor_source == "model":
			_model_failed("世界拒绝模型动作")
			return
		source_detail = "当前无法执行该动作（资源、距离或材料不足）。"
		_refresh_ui()
		return
	if lesson_id != "open":
		curriculum.after_action(world, action, actor_source)
	physics_world.sync_world(world)
	if paused and physics_world.is_moving():
		physics_world.complete_paused_step = true
	source = actor_source
	match actor_source:
		"baseline":
			source_detail = "规则基线 · 非 LLM / " + reason
		"model":
			source_detail = ("模型在线验收 / " if allow_model_validation else "模型决策 / ") + reason
		"fallback":
			source_detail = "降级等待 · 非模型成功 / " + reason
			fallback_count += 1
		"human":
			source_detail = "手动控制 / " + reason
	sample_count += 1
	last_action = _verb_name(str(action.verb))
	if str(action.verb) != "move" or world.tick % 8 == 0:
		_append_log("%03d  %s" % [world.tick, world.last_event])
	if session_log != null:
		session_log.store_line(JSON.stringify(_identity_record({"tick_before": before, "tick_after": world.tick, "source": actor_source, "action": action, "reason": reason, "health": world.health, "day": world.day(), "profile": world.profile,  "latency_ms": last_latency if actor_source == "model" else 0.0, "model_decision": last_model_decision if actor_source == "model" else {}, "runtime_status": _runtime_status(), "vitals": _runtime_vitals()})))
		session_log.flush()
	_play_effect("build_camp" if lesson_id != "open" and curriculum.state().status == "success" else str(action.verb))
	if _lesson_finished():
		paused = true
		physics_world.frozen = true
		physics_world.complete_paused_step = true
		held_direction = Vector2i.ZERO
	_update_music()
	_refresh_ui()

func _update_music() -> void:
	if music == null or DisplayServer.get_name() == "headless" or world.is_night() == music_is_night:
		return
	music_is_night = world.is_night()
	var path: String = "res://assets/audio/night_loop.wav" if music_is_night else "res://assets/audio/meadow_loop.wav"
	if ResourceLoader.exists(path):
		music.stop()
		music.stream = load(path)
		if not muted:
			music.play()

func _append_log(message: String) -> void:
	logs.push_front(message)
	if logs.size() > 4:
		logs.pop_back()

func _set_mode(value: String) -> void:
	current_model_error = ""
	if request_pending:
		http.cancel_request()
		request_pending = false
	balanced_health_pending = false
	request_epoch += 1
	mode = value
	connected_model = ""
	last_model_decision = {}
	last_latency = 0.0
	if value == "model" and sidebar != null and not observer_open:
		_toggle_observer()
	held_direction = Vector2i.ZERO
	source = "none" if value == "model" else "human" if value == "human" else "baseline"
	manual_target = Vector2i(-1, -1)
	manual_resource = ""
	timer = 0.0
	source_detail = "点击已探索陆地移动，点击资源采集。" if value == "human" else "本地规则基线 · 非 LLM" if value == "baseline" else "正在连接模型；尚未返回决策。"
	_refresh_ui()

func _reset_open_world(seed_value: int, difficulty: String) -> void:
	# Only an explicit profile selects experimental/released M7 rules. The
	# launcher supplies it only after a verified release updates its config.
	if difficulty == "open4":
		if not world is SurvivalWorld:
			world = SurvivalWorld.new()
	elif difficulty == "open3":
		if not world is ExplorationWorld:
			world = ExplorationWorld.new()
	elif difficulty == "open2":
		if not world is BalancedWorld:
			world = BalancedWorld.new()
	elif difficulty in ["m7_control", "m7_terrain"]:
		if not world is ChallengeWorld:
			world = ChallengeWorld.new()
	elif world is ChallengeWorld or world is BalancedWorld or world is ExplorationWorld or world is SurvivalWorld:
		world = World.new()
	world.reset(seed_value, difficulty)
	if physics_world != null:
		physics_world.world = world
	if world_view != null:
		world_view.world = world

func _restart() -> void:
	if request_pending:
		http.cancel_request()
	request_pending = false
	balanced_health_pending = false
	request_epoch += 1
	if lesson_id == "open":
		_reset_open_world(world.world_seed, open_difficulty)
	else:
		curriculum.configure(world, lesson_id, world.world_seed)
	physics_world.sync_world(world, true)
	physics_world.frozen = false
	held_direction = Vector2i.ZERO
	world_view.snap_camera()
	manual_target = Vector2i(-1, -1)
	manual_resource = ""
	logs.clear()
	sample_count = 0
	fallback_count = 0
	model_error_count = 0
	current_model_error = ""
	last_model_decision = {}
	last_action = "观察"
	last_latency = 0.0
	source = "baseline" if mode == "baseline" else "human" if mode == "human" else "none"
	source_detail = "新旅程 · 等待下一次决策"
	paused = false
	timer = 0.0
	_append_log("同一种子，新的旅程。")
	_update_music()
	_refresh_ui()

func _toggle_pause() -> void:
	if _lesson_finished():
		return
	paused = not paused
	current_model_error = ""
	physics_world.frozen = paused
	physics_world.complete_paused_step = false
	if paused and request_pending:
		http.cancel_request()
		request_pending = false
		source_detail = "已暂停，并取消在途请求。"
	if session_log != null:
		session_log.store_line(JSON.stringify(_identity_record({"event": "runtime_status", "tick": world.tick, "runtime_status": _runtime_status(), "vitals": _runtime_vitals(), "agent_context": _model_context()})))
		session_log.flush()
	_refresh_ui()

func _toggle_difficulty() -> void:
	if world.profile == "open4":
		_restart()
		return
	if lesson_id != "open":
		return
	open_difficulty = standard_difficulty if world.profile == "d0" else "d0"
	_restart()
	_append_log("已重开：" + ("轻松生存 · 消耗减半 / 无夜间伤害" if world.profile == "d0" else "探索生存 · 试验规则" if world.profile in ["open2", "open3", "open4"] else "D1 标准 · 完整昼夜与危险"))
	_refresh_ui()

func _toggle_collision() -> void:
	world_view.debug_collision = not world_view.debug_collision
	_refresh_ui()

func _cycle_speed() -> void:
	speed = 2 if speed == 1 else 4 if speed == 2 else 1
	_refresh_ui()

func _toggle_audio() -> void:
	muted = not muted
	if muted:
		music.stop()
		effects.stop()
	elif music.stream != null:
		music.play()
	_refresh_ui()

func _manual_action(verb: String) -> void:
	_set_mode("human")
	var action: Dictionary = {"verb": verb}
	if verb == "gather":
		for candidate in world.allowed_actions():
			if str(candidate.verb) == "gather":
				action = candidate
				break
	_execute(action, "human", "手动动作")

func _input(event: InputEvent) -> void:
	# A focused HUD button must not consume the paused world's single-step key.
	if paused and event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_SPACE:
		_step()
		get_viewport().set_input_as_handled()

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and not event.pressed and event.keycode in [KEY_W, KEY_UP, KEY_A, KEY_LEFT, KEY_S, KEY_DOWN, KEY_D, KEY_RIGHT]:
		held_direction = Vector2i.ZERO
	if event is InputEventMouseMotion:
		world_view.hover_tile = world_view.screen_to_tile(event.position)
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		if get_viewport_rect().has_point(event.position):
			var target: Vector2i = world_view.screen_to_tile(event.position)
			if world.discovered.has(world.tile_id(target)) and (world.is_walkable(target) or not world.resource_at(target).is_empty()):
				_set_mode("human")
				manual_target = target
				var resource: Dictionary = world.resource_at(target)
				manual_resource = str(resource.get("id", "")) if int(resource.get("ready_at", 0)) <= world.tick else ""
				world_view.marker = target
				world_view.marker_age = 0.0
				if paused:
					_step()
			else:
				source_detail = "目标为水面、建筑或未知区域，不能直接进入。"
				_refresh_ui()
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_F3:
				_toggle_collision()
			KEY_P:
				_toggle_pause()
			KEY_SPACE:
				if paused:
					_step()
			KEY_R:
				_restart()
			KEY_M:
				_toggle_audio()
			KEY_E:
				_manual_action("gather")
			KEY_F:
				_manual_action("eat")
			KEY_Q:
				_manual_action("drink")
			KEY_B:
				_manual_action("build_camp")
			KEY_T:
				_manual_action("rest")
			KEY_W, KEY_UP, KEY_A, KEY_LEFT, KEY_S, KEY_DOWN, KEY_D, KEY_RIGHT:
				var direction: Vector2i = Vector2i.UP
				if event.keycode in [KEY_A, KEY_LEFT]:
					direction = Vector2i.LEFT
				elif event.keycode in [KEY_S, KEY_DOWN]:
					direction = Vector2i.DOWN
				elif event.keycode in [KEY_D, KEY_RIGHT]:
					direction = Vector2i.RIGHT
				_set_mode("human")
				held_direction = direction
				_step()

func _build_ui() -> void:
	var canvas := CanvasLayer.new()
	add_child(canvas)
	ui_root = Control.new()
	ui_root.name = "SurvivalHUD"
	ui_root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	ui_root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	canvas.add_child(ui_root)
	var theme := Theme.new()
	var font := SystemFont.new()
	font.font_names = PackedStringArray(["Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", "sans-serif"])
	theme.default_font = font
	theme.default_font_size = 14
	theme.set_color("font_color", "Label", INK)
	theme.set_color("font_color", "Button", INK)
	theme.set_color("font_hover_color", "Button", ACCENT)
	theme.set_color("font_disabled_color", "Button", Color("a4ac9a"))
	theme.set_stylebox("normal", "Button", _box(Color("eff3e3"), 10, Color("dce4ce")))
	theme.set_stylebox("hover", "Button", _box(Color("dfeccf"), 10, Color("a5c599")))
	theme.set_stylebox("pressed", "Button", _box(Color("c4dab7"), 10, Color("8eaf83")))
	theme.set_stylebox("disabled", "Button", _box(Color("ebeddf"), 10, Color("e0e5d5")))
	theme.set_stylebox("focus", "Button", _box(Color(0, 0, 0, 0), 10, ACCENT))
	ui_root.theme = theme
	hud_frame = MarginContainer.new()
	hud_frame.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	hud_frame.mouse_filter = Control.MOUSE_FILTER_IGNORE
	for edge in ["left", "right", "top", "bottom"]:
		hud_frame.add_theme_constant_override("margin_" + edge, 20)
	ui_root.add_child(hud_frame)
	var layout := _column(hud_frame, 12)
	layout.mouse_filter = Control.MOUSE_FILTER_IGNORE
	hud_top = _card(layout)
	hud_top.name = "SurvivalStatus"
	var top := _row(hud_top, 24)
	var brand := _column(top, 0)
	brand.custom_minimum_size.x = 135
	title_label = _label("芽野", 28, INK)
	brand.add_child(title_label)
	brand.add_child(_label("MEADOWMIND", 10, MUTED))
	var vital_definitions: Array = [["health", "生命", "df8577"], ["hunger", "饱食", "d5a047"], ["thirst", "水分", "58aebe"]]
	if world.profile != "open4": vital_definitions.append(["energy", "体力", "81a963"])
	for def in vital_definitions:
		var meter := _column(top, 7)
		meter.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		meter.size_flags_vertical = Control.SIZE_SHRINK_CENTER
		meter.custom_minimum_size.x = 115
		var row := _row(meter, 4)
		var caption := _label(str(def[1]), 13, MUTED)
		caption.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		row.add_child(caption)
		var value := _label("100", 16, INK)
		row.add_child(value)
		vital_labels[str(def[0])] = value
		var bar := _meter(meter, Color(str(def[2])), 8)
		vital_bars[str(def[0])] = bar
	var time_box := _column(top, 7)
	time_box.custom_minimum_size.x = 146
	time_box.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	phase_label = _label("第 1 天 · 晨光", 16, INK)
	time_box.add_child(phase_label)
	day_bar = _meter(time_box, Color("d8b66a"), 5)
	day_bar.max_value = world.DAY_STEPS
	observer_button = _button("观察面板  ≡", _toggle_observer)
	observer_button.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	observer_button.custom_minimum_size.y = 42
	top.add_child(observer_button)
	var field := Control.new()
	field.name = "UnobstructedWorld"
	field.mouse_filter = Control.MOUSE_FILTER_IGNORE
	field.size_flags_vertical = Control.SIZE_EXPAND_FILL
	layout.add_child(field)
	sidebar = _card(field)
	sidebar.name = "ObserverPanel"
	sidebar.custom_minimum_size.x = 300
	sidebar.set_anchors_and_offsets_preset(Control.PRESET_TOP_RIGHT)
	sidebar.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	var side := _column(sidebar, 10)
	side.add_child(_label("野外观察", 20, INK))
	source_label = _label("", 13, MUTED)
	source_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	source_label.custom_minimum_size.x = 264
	side.add_child(source_label)
	latency_label = _label("", 11, MUTED)
	side.add_child(latency_label)
	side.add_child(_separator())
	progress_label = _label("", 12, MUTED)
	side.add_child(progress_label)
	side.add_child(_label("最近发生", 12, MUTED))
	log_label = _label("", 13, INK)
	log_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	log_label.custom_minimum_size = Vector2(264, 90)
	side.add_child(log_label)
	side.add_child(_separator())
	var settings := _row(side, 6)
	difficulty_button = _button("标准生存", _toggle_difficulty)
	difficulty_button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	difficulty_button.tooltip_text = "切换难度会重开当前世界"
	settings.add_child(difficulty_button)
	settings.add_child(_button("碰撞 F3", _toggle_collision))
	collision_label = _label("", 11, MUTED)
	side.add_child(collision_label)
	sidebar.visible = observer_open
	inventory_panel = _card(field)
	inventory_panel.name = "Backpack"
	inventory_panel.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_LEFT)
	inventory_panel.grow_vertical = Control.GROW_DIRECTION_BEGIN
	var pocket := _column(inventory_panel, 5)
	pocket.add_child(_label("随身行囊", 11, MUTED))
	inventory_label = _label("", 16, INK)
	pocket.add_child(inventory_label)
	camp_label = _label("", 11, MUTED)
	pocket.add_child(camp_label)
	var dock := _column(layout, 8)
	dock.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var action_center := CenterContainer.new()
	action_center.mouse_filter = Control.MOUSE_FILTER_IGNORE
	dock.add_child(action_center)
	var action_row := _row(_card(action_center), 6)
	for def in [["gather", "采集", "E"], ["eat", "进食", "F"], ["drink", "饮水", "Q"], ["build_camp", "建营", "B"], ["rest", "休息", "T"], ["wait", "等待", "·"]]:
		var button := _button(str(def[1]) + "  " + str(def[2]), func() -> void: _manual_action(str(def[0])))
		button.custom_minimum_size = Vector2(82, 42)
		button.set_meta("verb", str(def[0]))
		button.tooltip_text = "执行后切换为手动控制；灰色表示当前条件不满足"
		action_row.add_child(button)
		context_buttons.append(button)
	hud_bottom = _card(dock)
	hud_bottom.name = "ControlDock"
	var bottom := _row(hud_bottom, 12)
	var modes := _row(bottom, 4)
	for pair in [["human", "手动"], ["model", "模型接管"], ["baseline", "规则演示"]]:
		var button := _button(str(pair[1]), func() -> void: _set_mode(str(pair[0])))
		button.custom_minimum_size.y = 42
		modes.add_child(button)
		mode_buttons[str(pair[0])] = button
	var status := _column(bottom, 2)
	status.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	status.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	mode_label = _label("", 12, ACCENT)
	mode_label.clip_text = true
	mode_label.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	status.add_child(mode_label)
	event_label = _label("", 12, MUTED)
	event_label.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	event_label.clip_text = true
	status.add_child(event_label)
	var replay_button := _button("历史回放", _open_replay)
	replay_button.name = "HistoricalReplay"
	replay_button.disabled = not FileAccess.file_exists("res://output/replays/current.json")
	replay_button.tooltip_text = "暂无已保存的回放" if replay_button.disabled else "结束当前会话，观看已保存的回放；返回后开始新世界"
	bottom.add_child(replay_button)
	pause_button = _button("暂停 P", _toggle_pause)
	bottom.add_child(pause_button)
	speed_button = _button("1×", _cycle_speed)
	bottom.add_child(speed_button)
	sound_button = _button("声音 开", _toggle_audio)
	bottom.add_child(sound_button)
	bottom.add_child(_button("重开 R", _restart))
	# Retain historical CLI fixtures without exposing a teaching flow in the game.
	legacy_course_ui = Control.new()
	legacy_course_ui.name = "LegacyEvaluationOnly"
	legacy_course_ui.hide()
	ui_root.add_child(legacy_course_ui)
	lesson_select = OptionButton.new()
	lesson_select.add_item("开放世界")
	for item in Curriculum.metadata():
		lesson_select.add_item(str(item.title))
	lesson_select.item_selected.connect(_select_lesson)
	legacy_course_ui.add_child(lesson_select)
	lesson_next = _button("下一课", _next_lesson)
	legacy_course_ui.add_child(lesson_next)
	lesson_label = _label("", 12, INK)
	lesson_initial = _label("", 12, INK)
	lesson_ledger = _label("", 12, INK)
	lesson_progress = _label("", 12, INK)
	for label in [lesson_label, lesson_initial, lesson_ledger, lesson_progress]:
		legacy_course_ui.add_child(label)

func _open_replay() -> void:
	paused = true
	if request_pending:
		http.cancel_request()
	request_pending = false
	balanced_health_pending = false
	request_epoch += 1
	if session_log != null:
		session_log.flush()
		session_log.close()
	get_tree().change_scene_to_file("res://scenes/replay.tscn")

func _toggle_observer() -> void:
	observer_open = not observer_open
	sidebar.visible = observer_open
	observer_button.text = "收起观察  ×" if observer_open else "观察面板  ≡"

func _card(parent: Node) -> PanelContainer:
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", _box(Color(0.98, 0.985, 0.94, 0.95), 16, Color("d9e2ca")))
	parent.add_child(panel)
	return panel

func _row(parent: Node, gap: int) -> HBoxContainer:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", gap)
	parent.add_child(row)
	return row

func _column(parent: Node, gap: int) -> VBoxContainer:
	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation", gap)
	parent.add_child(column)
	return column

func _meter(parent: Node, color: Color, height: int) -> ProgressBar:
	var bar := ProgressBar.new()
	bar.show_percentage = false
	bar.custom_minimum_size.y = height
	for item in [["background", Color("e4e9d8")], ["fill", color]]:
		var box := StyleBoxFlat.new()
		box.bg_color = item[1]
		box.set_corner_radius_all(4)
		bar.add_theme_stylebox_override(str(item[0]), box)
	parent.add_child(bar)
	return bar

func _layout() -> void:
	world_view.view_size = get_viewport_rect().size

func _refresh_ui() -> void:
	if phase_label == null:
		return
	var phase_names: Dictionary = {"dawn": "晨光", "day": "晴昼", "dusk": "暮色", "night": "星夜"}
	phase_label.text = "第 %d 天 · %s" % [world.day(), str(phase_names[world.phase()])]
	for key in vital_bars:
		var value: float = float(world.get(str(key)))
		vital_bars[key].value = value
		vital_labels[key].text = "%d" % roundi(value)
	var runtime: Dictionary = _runtime_status()
	var model_status: String = "生命耗尽" if world.dead else "连接异常 · 已暂停" if runtime.state == "model_error" else "正在思考" if request_pending else "用户暂停" if paused else "真实模型决策" if source == "model" else "等待首个决策"
	mode_label.text = "手动生存 · WASD 移动 / 点击采集" if mode == "human" else "规则演示 · 非 LLM" if mode == "baseline" else "模型接管 · " + model_status
	if world.profile in ["open2", "open3", "open4"]:
		mode_label.text = "探索生存 · 试验规则 · " + mode_label.text
	source_label.text = ("模型：" + connected_model + "\n" if not connected_model.is_empty() else "") + source_detail
	event_label.text = str(runtime.death_detail) + " · 按 R 重开" if world.dead else "模型连接异常 · 世界已暂停，按 P 重试" if runtime.state == "model_error" else "用户暂停 · 按 P 继续" if paused else last_action + "  /  " + world.last_event
	inventory_label.text = "木材 %d    石块 %d    野果 %d" % [int(world.inventory.wood), int(world.inventory.stone), int(world.inventory.berries)]
	camp_label.text = "营地已建立 · 附近休息可恢复生命" if bool(world.camp.exists) else "营地材料  %d / 4 木材  ·  %d / 2 石块" % [int(world.inventory.wood), int(world.inventory.stone)]
	progress_label.text = "已探索 %d 格  ·  世界步 %d · 无天数/步数上限" % [world.discovered.size(), world.tick]
	day_bar.value = world.tick % world.DAY_STEPS
	log_label.text = "\n\n".join(logs.slice(0, 3))
	latency_label.text = "决策 %d 次 · 异常 %d 次%s" % [sample_count, model_error_count, " · %.0f ms" % last_latency if last_latency > 0.0 else ""]
	difficulty_button.disabled = lesson_id != "open"
	_update_lesson_ui()
	difficulty_button.text = "轻松生存 · 重开" if world.profile == "d0" else "地形疲劳 · 标准" if world.profile == "m7_terrain" else "探索生存 · 试验规则" if world.profile in ["open2", "open3", "open4"] else "标准生存 · 重开"
	if world.profile in ["open2", "open3", "open4"]:
		progress_label.text += "\n探索生存 · 试验规则 · AI尚未通过验收"
	if world.profile in ["open3", "open4"]:
		progress_label.text += "\n每步水耗 %.2f · 饥饿消耗 %.2f" % [world.balance.thirst_drain, world.balance.hunger_drain]
	if world.profile == "open4":
		progress_label.text += "\n三项生存数值 · 夜晚本身不造成伤害"
		difficulty_button.text = "当前规则 · 重开"
		difficulty_button.tooltip_text = "使用当前三项生存规则重开同一种子"
		if bool(world.camp.exists): camp_label.text = "营地已建立 · 食水均至少20时，附近休息恢复2生命"
	if world.profile == "m7_terrain":
		progress_label.text += "\n行走体力：林地 0.70 · 沙地 1.00"
	collision_label.text = "F3 碰撞已显示 · 红色实体 / 蓝色脚底" if world_view.debug_collision else "F3 查看脚底碰撞"
	pause_button.text = "继续 P" if paused else "暂停 P"
	speed_button.text = "%d×" % speed
	sound_button.text = "声音 关" if muted else "声音 开"
	for key in mode_buttons:
		mode_buttons[key].add_theme_color_override("font_color", ACCENT if str(key) == mode else MUTED)
		mode_buttons[key].add_theme_stylebox_override("normal", _box(Color("d9eacb") if str(key) == mode else Color("f2f6e8"), 12, Color("acc99f") if str(key) == mode else Color("dfe8d4")))
	var legal_verbs: Array[String] = []
	for action in world.allowed_actions():
		legal_verbs.append(str(action.verb))
	for button in context_buttons:
		button.disabled = _lesson_finished() or not legal_verbs.has(str(button.get_meta("verb")))

func _verb_name(verb: String) -> String:
	return str({"move": "探索", "navigate": "导航", "gather": "采集", "eat": "进食", "drink": "饮水", "rest": "休息", "build_camp": "建立营地", "wait": "观察"}.get(verb, verb))

func _label(text: String, font_size: int, color: Color) -> Label:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	return label

func _button(text: String, callback: Callable) -> Button:
	var button := Button.new()
	button.text = text
	button.focus_mode = Control.FOCUS_ALL
	button.pressed.connect(callback)
	return button

func _box(color: Color, radius: int, border: Color = Color.TRANSPARENT) -> StyleBoxFlat:
	var box := StyleBoxFlat.new()
	box.bg_color = color
	box.set_corner_radius_all(radius)
	box.border_color = border
	box.set_border_width_all(1 if border.a > 0.0 else 0)
	box.content_margin_left = 12
	box.content_margin_right = 12
	box.content_margin_top = 9
	box.content_margin_bottom = 9
	# Shadows stay low contrast and limited to UI panels.
	if radius >= 16:
		box.shadow_color = Color(0.20, 0.34, 0.23, 0.08)
		box.shadow_size = 8
		box.shadow_offset = Vector2(0, 3)
	return box

func _separator() -> HSeparator:
	var separator := HSeparator.new()
	var style := StyleBoxLine.new()
	style.color = Color("e2e9d8")
	style.thickness = 1
	separator.add_theme_stylebox_override("separator", style)
	return separator

func _lesson_finished() -> bool:
	return lesson_id != "open" and str(curriculum.state().get("status", "active")) != "active"

func _select_lesson(index: int) -> void:
	if world.profile == "open4": return
	lesson_id = "open" if index == 0 else str(Curriculum.metadata()[index - 1].id)
	_restart()

func _next_lesson() -> void:
	_select_lesson(1 if lesson_select.selected == 0 else mini(6, lesson_select.selected + 1))

func _update_lesson_ui() -> void:
	var selected: int = 0
	for index in range(Curriculum.metadata().size()):
		if str(Curriculum.metadata()[index].id) == lesson_id:
			selected = index + 1
	lesson_select.select(selected)
	lesson_next.disabled = selected == 6
	if lesson_id == "open":
		lesson_label.text = "自由生存：找到食物与水，收集材料，建立营地"
		lesson_progress.text = "开放世界 · 可切换上方教学任务"
		lesson_initial.text = "程序生成的完整世界 · 无课程初始条件"
		lesson_ledger.text = "探索与生存由当前控制者决定"
		return
	var task: Dictionary = curriculum.state()
	lesson_label.text = str(task.goal) + " · 给定目标练习 / D0 受控起点"
	var status_names: Dictionary = {"active": "进行中", "success": "完成！", "failed": "未完成 · 步预算用尽或生命耗尽"}
	var source_names: Dictionary = {"none": "尚无动作", "model": "模型", "baseline": "规则", "human": "手动", "mixed": "混合", "fallback": "降级等待"}
	lesson_progress.text = "%s · %d / %d 世界步 · 来源 %s" % [status_names[task.status], task.elapsed, task.budget_world_steps, source_names.get(str(task.source), str(task.source))]
	var initial_inventory: Dictionary = task.initial.inventory
	lesson_initial.text = "受控初值：食物 %d · 木 %d · 石 %d · 饱食 %d · 水分 %d" % [initial_inventory.berries, initial_inventory.wood, initial_inventory.stone, task.initial.hunger, task.initial.thirst]
	var progress: Dictionary = task.progress
	var achieved: bool = str(task.status) == "success"
	match lesson_id:
		"forage":
			lesson_ledger.text = "已采食物 %d 份 · 吃食%s" % [progress.food_gathered, "已达成" if achieved else "未达成"]
		"camp_chain":
			lesson_ledger.text = "累计采集：木 %d / 4 · 石 %d / 2 · 营地%s" % [progress.wood_gathered, progress.stone_gathered, "已完成" if world.camp.exists else "未完成"]
		"camp_ready":
			lesson_ledger.text = "营地%s · 当前材料：木 %d · 石 %d" % ["已完成" if world.camp.exists else "未完成", world.inventory.wood, world.inventory.stone]
		"fetch_water":
			lesson_ledger.text = "实际移动 %d 步 · 饮水%s" % [progress.moves, "已达成" if achieved else "未达成"]
		"drink":
			lesson_ledger.text = "饮水%s · 当前水分 %d" % ["已达成" if achieved else "未达成", world.thirst]
		"eat":
			lesson_ledger.text = "吃食%s · 背包食物 %d" % ["已达成" if achieved else "未达成", world.inventory.berries]
	if _lesson_finished():
		event_label.text = str(status_names[task.status]) + " · 可重试 / 下一课"
