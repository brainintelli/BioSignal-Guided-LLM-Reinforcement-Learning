extends Node2D
## Independent historical replay viewer: deliberately no HTTPRequest or live controller.
const Replay = preload("res://scripts/replay_session.gd")
const Physics = preload("res://scripts/physics_world.gd")
const View = preload("res://scripts/world_view.gd")
const ReplayAudio = preload("res://scripts/replay_audio.gd")
const DEFAULT_FILE: String = "res://output/replays/current.json"
@export var replay_path: String = DEFAULT_FILE
var session = Replay.new()
var physics: Node2D
var view: Node2D
var playing: bool = false
var speed: int = 1
var timer: float = 0.0
var seek_target: int = -1
var status: Label
var vitals: Label
var counter: Label
var play_button: Button
var speed_button: Button
var sound_button: Button
var audio: Node
var slider: HSlider
var scrub_updating: bool = false
var scrub_dragging: bool = false
var elapsed: float = 0.0
var smoke_seconds: float = 0.0
var screenshot_path: String = ""
var leaving: bool = false
var previous_auto_accept_quit: bool = true

static func provenance_text(data: Dictionary) -> String:
	if data.is_empty():
		return "来源载入失败"
	var label: String = str(data.get("source_label", "M6 正式独立测试"))
	var ending: String = "达到记录上限，仍存活" if bool(data.right_censored) else "自然死亡"
	return "%s | seed %d · rep %d | v%d / %s\n%s · 本来源 %d 局 / 已验证比较范围 %d 局\n记录上限 %d 步 · 最长已观测 %d 步（%s）· %d 局并列，按 seed、rep 展示。" % [data.model, int(data.seed), int(data.replicate), int(data.world_version), str(data.profile), label, int(data.source_episode_count), int(data.get("selection_scope_episode_count", data.source_episode_count)), int(data.horizon), int(data.display_end_tick), ending, int(data.tied_longest_count)]

static func endpoint_text(data: Dictionary) -> String:
	if bool(data.right_censored):
		return "记录结束：%d 步时仍存活（右删失）；不能推断之后存活多久。" % int(data.display_end_tick)
	return "记录结束：第 %d 步真实死亡；记录上限为 %d 步。" % [int(data.display_end_tick), int(data.horizon)]

func _ready() -> void:
	previous_auto_accept_quit = get_tree().auto_accept_quit
	get_tree().auto_accept_quit = false
	var path: String = replay_path
	var initial_seek: int = 0
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--replay-file="):
			path = argument.trim_prefix("--replay-file=")
		elif argument == "--replay-autoplay":
			playing = true
		elif argument.begins_with("--replay-seek="):
			initial_seek = int(argument.trim_prefix("--replay-seek="))
		elif argument.begins_with("--replay-speed="):
			var requested_speed: int = int(argument.trim_prefix("--replay-speed="))
			speed = requested_speed if requested_speed in [1, 2, 4, 8] else 1
		elif argument.begins_with("--replay-smoke-seconds="):
			smoke_seconds = float(argument.trim_prefix("--replay-smoke-seconds="))
		elif argument.begins_with("--replay-screenshot="):
			screenshot_path = argument.trim_prefix("--replay-screenshot=")
	var loaded: bool = session.load_file(path)
	if not loaded:
		# Only initialize a hidden renderer shell; no substitute replay is executed.
		session.engine.world.reset(2304, "open1")
	physics = Physics.new()
	physics.world = session.engine.world
	add_child(physics)
	view = View.new()
	view.world = session.engine.world
	view.physics_world = physics
	add_child(view)
	view.visible = loaded
	audio = ReplayAudio.new()
	add_child(audio)
	_build_ui()
	get_viewport().size_changed.connect(_resize)
	_resize()
	if loaded:
		view.snap_camera()
		if initial_seek > 0:
			_seek(initial_seek)
	else:
		playing = false
	_refresh()

func _process(delta: float) -> void:
	if leaving:
		return
	elapsed += delta
	if smoke_seconds > 0.0 and elapsed >= smoke_seconds and seek_target < 0:
		smoke_seconds = 0.0
		_finish_smoke()
	if not session.error.is_empty():
		playing = false
		audio.stop()
		physics.frozen = true
		view.visible = false
		return
	if seek_target >= 0:
		var started: int = Time.get_ticks_usec()
		var steps: int = 0
		while session.cursor < seek_target and steps < 24 and Time.get_ticks_usec() - started < 6000:
			if not session.advance():
				seek_target = -1
				break
			steps += 1
		if session.cursor >= seek_target or not session.error.is_empty():
			seek_target = -1
			_bind_world(true)
		_refresh()
		return
	physics.frozen = not playing
	physics.movement_speed = 230.0 * speed
	if physics.unexpected_collisions > 0:
		session.error = "回放碰撞与原始路径不一致，已停止"
		_refresh()
		return
	if playing and not session.finished:
		timer += delta
		if timer >= 0.32 / speed and not physics.is_moving():
			timer = 0.0
			_step()
	if playing and session.finished and not physics.is_moving():
		playing = false
		_refresh()

func _step() -> void:
	if leaving:
		return
	if not playing:
		audio.stop()
	if seek_target >= 0 or not session.error.is_empty() or session.finished:
		return
	if physics.is_moving():
		physics.complete_paused_step = true
		return
	if session.advance():
		_bind_world(false)
		physics.complete_paused_step = true
	_refresh()
	if playing and not session.finished and session.error.is_empty():
		audio.play_effect(str(session.data.records[session.cursor - 1].action.verb))

func _bind_world(reset_actor: bool) -> void:
	session.apply_presentation_boundary()
	view.visible = true
	physics.sync_world(session.engine.world, reset_actor)
	view.world = session.engine.world
	if reset_actor:
		view.snap_camera()

func _seek(index: int) -> void:
	if leaving or session.data.is_empty():
		return
	playing = false
	audio.stop()
	physics.frozen = true
	physics.complete_paused_step = false
	if not session.restart():
		_refresh()
		return
	seek_target = clampi(index, 0, session.data.records.size())
	# Keep the previous verified frame visible until the reconstructed world is ready.
	if seek_target == 0:
		seek_target = -1
		_bind_world(true)
	_refresh()

func _toggle() -> void:
	if leaving or session.finished or not session.error.is_empty() or seek_target >= 0:
		return
	playing = not playing
	physics.complete_paused_step = false
	_refresh()

func _toggle_sound() -> void:
	audio.muted = not audio.muted
	_refresh()

func _sync_audio() -> void:
	if audio == null:
		return
	var active: bool = playing and seek_target < 0 and not session.finished and session.error.is_empty()
	audio.sync(active, session.engine.world.is_night() if not session.data.is_empty() else false)
	if sound_button != null:
		sound_button.text = "声音：关" if audio.muted else "声音：开"

func _exit_tree() -> void:
	get_tree().auto_accept_quit = previous_auto_accept_quit
	if is_instance_valid(audio):
		audio.stop()

func _prepare_exit() -> void:
	leaving = true
	playing = false
	physics.frozen = true
	physics.complete_paused_step = false
	audio.stop()
	# Stop before freeing players; the mixer releases WAV voices asynchronously.
	await get_tree().create_timer(0.06).timeout
	await get_tree().process_frame

func _leave(to_survival: bool) -> void:
	if leaving:
		return
	await _prepare_exit()
	if to_survival:
		get_tree().change_scene_to_file("res://scenes/main.tscn")
	else:
		get_tree().quit()

func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST and is_inside_tree():
		_leave(false)

func _speed() -> void:
	speed = speed * 2 if speed < 8 else 1
	speed_button.text = "%d×" % speed

func _resize() -> void:
	view.view_size = get_viewport_rect().size

func _refresh() -> void:
	_sync_audio()
	play_button.text = "暂停" if playing else "播放"
	if not session.error.is_empty():
		status.text = "校验失败 · " + session.error
		status.modulate = Color("b44839")
		playing = false
		return
	if session.data.is_empty():
		return
	var world = session.engine.world
	vitals.text = "生命 %.1f   饱食 %.1f   水分 %.1f   体力 %.1f    木 %d / 石 %d / 食物 %d" % [world.health, world.hunger, world.thirst, world.energy, world.inventory.wood, world.inventory.stone, world.inventory.berries]
	counter.text = "世界步 %d / %d · 决策 %d / %d" % [session.display_tick(), int(session.data.display_end_tick), session.cursor, session.data.records.size()]
	if seek_target >= 0:
		status.text = "正在逐步校验并跳转 · %d / %d" % [session.cursor, seek_target]
	elif session.finished:
		status.text = endpoint_text(session.data)
	else:
		var action: String = str(session.data.records[maxi(0, session.cursor - 1)].action.verb) if session.cursor > 0 else "初始状态"
		status.text = "逐步状态校验通过 · " + action + (" · 播放中" if playing else " · 已暂停")
	if not scrub_dragging:
		scrub_updating = true
		slider.value = session.cursor
		scrub_updating = false

func _button(text_value: String, callback: Callable) -> Button:
	var button := Button.new()
	button.text = text_value
	button.custom_minimum_size = Vector2(92, 42)
	button.pressed.connect(callback)
	return button

func _panel() -> PanelContainer:
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color("f9fbf0")
	style.corner_radius_top_left = 16
	style.corner_radius_top_right = 16
	style.corner_radius_bottom_left = 16
	style.corner_radius_bottom_right = 16
	style.content_margin_left = 18
	style.content_margin_right = 18
	style.content_margin_top = 12
	style.content_margin_bottom = 12
	panel.add_theme_stylebox_override("panel", style)
	return panel

func _build_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var ui := Control.new()
	var theme := Theme.new()
	theme.default_font_size = 17
	for kind in ["Label", "Button"]:
		theme.set_color("font_color", kind, Color("2b4d42"))
	theme.set_color("font_hover_color", "Button", Color("245d43"))
	theme.set_color("font_pressed_color", "Button", Color("1f6747"))
	for state in ["normal", "hover", "pressed", "focus"]:
		var button_style := StyleBoxFlat.new()
		button_style.bg_color = Color("e6eedb") if state == "normal" else Color("d6e9c9")
		button_style.set_corner_radius_all(9)
		button_style.set_border_width_all(1)
		button_style.border_color = Color("b7cda5")
		theme.set_stylebox(state, "Button", button_style)
	ui.theme = theme
	ui.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	ui.mouse_filter = Control.MOUSE_FILTER_IGNORE
	ui.add_theme_color_override("font_color", Color("2b4d42"))
	ui.add_theme_font_size_override("font_size", 17)
	layer.add_child(ui)
	var top: PanelContainer = _panel()
	ui.add_child(top)
	top.set_anchors_and_offsets_preset(Control.PRESET_TOP_WIDE)
	top.offset_left = 20
	top.offset_right = -20
	top.offset_top = 16
	var header := VBoxContainer.new()
	top.add_child(header)
	var title := Label.new()
	title.text = "芽野 · 历史回放，不是实时模型"
	title.add_theme_font_size_override("font_size", 26)
	header.add_child(title)
	var provenance := Label.new()
	provenance.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	provenance.text = provenance_text(session.data)
	header.add_child(provenance)
	vitals = Label.new()
	vitals.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	header.add_child(vitals)
	var bottom: PanelContainer = _panel()
	ui.add_child(bottom)
	bottom.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_WIDE)
	bottom.grow_vertical = Control.GROW_DIRECTION_BEGIN
	bottom.offset_left = 20
	bottom.offset_right = -20
	bottom.offset_bottom = -16
	var controls := VBoxContainer.new()
	bottom.add_child(controls)
	status = Label.new()
	status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	controls.add_child(status)
	counter = Label.new()
	controls.add_child(counter)
	slider = HSlider.new()
	slider.min_value = 0
	slider.max_value = maxi(1, session.data.get("records", []).size())
	slider.step = 1
	slider.custom_minimum_size.y = 28
	slider.tooltip_text = "按决策边界跳转；逐步校验重建，不直接伪造中间状态"
	slider.drag_started.connect(func() -> void: scrub_dragging = true)
	slider.drag_ended.connect(func(changed: bool) -> void:
		scrub_dragging = false
		if changed:
			_seek(int(slider.value)))
	slider.value_changed.connect(func(value: float) -> void:
		if not scrub_updating and not scrub_dragging:
			_seek(int(value)))
	controls.add_child(slider)
	var buttons := HBoxContainer.new()
	buttons.add_theme_constant_override("separation", 12)
	controls.add_child(buttons)
	play_button = _button("播放", _toggle)
	buttons.add_child(play_button)
	buttons.add_child(_button("单步", func() -> void:
		playing = false
		_step()))
	speed_button = _button("%d×" % speed, _speed)
	buttons.add_child(speed_button)
	buttons.add_child(_button("从头", func() -> void: _seek(0)))
	buttons.add_child(_button("记录终点", func() -> void: _seek(session.data.get("records", []).size())))
	sound_button = _button("声音：开", _toggle_sound)
	buttons.add_child(sound_button)
	buttons.add_child(_button("返回生存", func() -> void: _leave(true)))

func _finish_smoke() -> void:
	await _prepare_exit()
	await RenderingServer.frame_post_draw
	if not screenshot_path.is_empty():
		get_viewport().get_texture().get_image().save_png(screenshot_path)
	print("REPLAY_SMOKE cursor=", session.cursor, " tick=", session.display_tick(), " finished=", session.finished, " error=", session.error)
	get_tree().quit(0 if session.error.is_empty() else 1)
