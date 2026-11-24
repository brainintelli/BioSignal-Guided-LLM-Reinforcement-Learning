extends SceneTree
## Synthetic boundary fixtures exercise real engine rules, not model performance.
const Session = preload("res://scripts/m8s3_environment_session.gd")
const MainScene = preload("res://scenes/main.tscn")
var failures: int = 0
var assertions: int = 0

func check(condition: bool, message: String) -> void:
	assertions += 1
	if not condition:
		failures += 1
		push_error(message)

func no_retired_fields(value: Variant) -> bool:
	if value is Dictionary:
		for key in value:
			if str(key) in ["energy", "exhaustion", "night_exposure"] or not no_retired_fields(value[key]):
				return false
	elif value is Array:
		for item in value:
			if not no_retired_fields(item): return false
	return true

func _initialize() -> void:
	call_deferred("run")

func run() -> void:
	var session = Session.new()
	# An opt-in in-memory negative control, never a modification of game files.
	if "--mutation-probe" in OS.get_cmdline_user_args():
		var mutant := GDScript.new()
		mutant.source_code = "extends \"res://scripts/survival_world.gd\"\nfunc _advance(steps: int) -> void:\n\tsuper._advance(steps)\n\tif is_night(): health -= 0.1\n"
		check(mutant.reload() == OK, "Negative-control script compiles")
		session.world = mutant.new()
	var first: Dictionary = session.handle({"op":"reset", "seed":2304, "profile":"open4"})
	check(first.ok, "New rules reset succeeds")
	if not first.ok: quit(1); return
	check(first.observation.profile == "open4" and first.observation.world_version == 6, "New identity is explicit")
	check(not session.handle({"op":"reset", "seed":2304, "profile":"open3"}).ok, "Old identity cannot silently enter new rules")
	session.handle({"op":"reset", "seed":2304, "profile":"open4"})
	var world = session.world
	check(no_retired_fields(first) and no_retired_fields(session.snapshot()), "No retired reserve in observation or exact snapshot")
	for verb in ["move", "navigate"]:
		var found: bool = false
		for action in world.allowed_actions():
			if action.verb != verb: continue
			var origin: Vector2i = world.player
			var reply: Dictionary = session.handle({"op":"step", "action":action})
			check(reply.ok and reply.transition.elapsed_ticks == 1, verb + " consumes exactly one tick")
			var offset: Vector2i = world.player - origin
			check(absi(offset.x) + absi(offset.y) == 1, verb + " moves exactly one cardinal tile")
			check(no_retired_fields(reply), "No energy in move history or trace")
			found = true
			break
		check(found, "Fixture must exercise " + verb)
	world.reset(2304, "open4")
	world.player = Vector2i(10, 10)
	world.tick = 139
	world.health = 80.0
	var night: Dictionary = session.handle({"op":"step", "action":{"verb":"wait"}})
	check(night.ok and world.is_night(), "Fixture crosses dusk into night")
	check(is_equal_approx(world.health, 80.0), "Night outside camp has no ambient damage")
	check(is_equal_approx(world.hunger, 81.8) and is_equal_approx(world.thirst, 77.92), "Food and water drain remain unchanged")
	world.hunger = 0.0
	world.thirst = 0.0
	var hungry: Dictionary = session.handle({"op":"step", "action":{"verb":"wait"}})
	check(is_equal_approx(world.health, 77.0), "Starvation and dehydration still deal 1.2 + 1.8")
	check(is_equal_approx(float(hungry.transition.trace[0].damage), 3.0), "Damage trace exposes exact health loss")
	world.reset(2304, "open4")
	world.player = world._layout_point(Vector2i(51, 47))
	world.tick = 145
	world.health = 80.0
	check(not world.visible_threats().is_empty(), "Threat fixture contains a visible enemy")
	session.handle({"op":"step", "action":{"verb":"wait"}})
	check(is_equal_approx(world.health, 78.8), "Visible nearby enemy damage remains 1.2, without added night damage")
	for eligible in [false, true]:
		world.reset(2304, "open4")
		world.health = 50.0
		world.camp = {"exists":true,"x":world.player.x+1,"y":world.player.y}
		world.hunger = 20.0 if eligible else 19.9
		world.thirst = 20.0
		var rest: Dictionary = session.handle({"op":"step", "action":{"verb":"rest"}})
		check(rest.ok and rest.transition.elapsed_ticks == 3, "Rest consumes three ticks")
		check(is_equal_approx(world.health, 52.0 if eligible else 50.0), "Camp healing uses food/water at action start")
		check(no_retired_fields(rest), "Rest has no energy delta or trace")
	world.reset(2304, "open4")
	world.health = 50.0
	session.handle({"op":"step", "action":{"verb":"rest"}})
	check(is_equal_approx(world.health, 50.0), "Rest outside camp does not heal")
	world.reset(2304, "open4")
	var resource_position: Vector2i = world.player + Vector2i.RIGHT
	world._remove_resource_at(resource_position)
	world._add_resource(resource_position, "berry")
	world._reveal()
	var gathered: Dictionary = session.handle({"op":"step", "action":{"verb":"gather", "target_id":"r_%d_%d" % [resource_position.x, resource_position.y]}})
	check(gathered.ok and gathered.transition.elapsed_ticks == 2 and world.inventory.berries == 5, "Gather preserves yield and duration")
	check(no_retired_fields(gathered), "Gather has no retired reserve anywhere")
	# The inherited legacy slot exists only for base-class compatibility. It must
	# not influence any current-rule transition, including terminal conditions.
	var low_legacy = Session.new()
	var high_legacy = Session.new()
	low_legacy.handle({"op":"reset", "seed":2304, "profile":"open4"})
	high_legacy.handle({"op":"reset", "seed":2304, "profile":"open4"})
	low_legacy.world.energy = 0.0
	high_legacy.world.energy = 100.0
	for verb in ["move", "navigate", "rest", "wait"]:
		var selected: Dictionary = {}
		for action in low_legacy.world.allowed_actions():
			if action.verb == verb:
				selected = action
				break
		check(not selected.is_empty(), "Legacy-slot probe has legal " + verb)
		var low_reply: Dictionary = low_legacy.handle({"op":"step", "action":selected})
		var high_reply: Dictionary = high_legacy.handle({"op":"step", "action":selected})
		check(low_reply == high_reply, "Legacy energy cannot affect current transition: " + verb)
	var scene = MainScene.instantiate()
	root.add_child(scene)
	await process_frame
	scene.set_process(false)
	check(scene.world.profile == "open4" and scene.world.observation().world_version == 6, "Default game runs new rules")
	check(scene.mode == "human", "Default game never starts model decisions")
	check(scene.vital_bars.size() == 3 and not scene.vital_bars.has("energy"), "HUD contains exactly three reserve meters")
	check(no_retired_fields(scene._runtime_vitals()), "Game runtime log exposes only three reserves")
	check(not scene._balanced_identity_matches({"profile":"open3","world_version":5,"environment_version":"m8_exploration_v1"}), "Old model identity is rejected")
	scene._restart()
	check(scene.world.profile == "open4" and scene.world.tick == 0, "Restart keeps new rules")
	scene._toggle_difficulty()
	check(scene.world.profile == "open4" and scene.vital_bars.size() == 3, "Current-rule button cannot bring back legacy stamina rules")
	check(not scene.progress_label.text.contains("体力"), "Current help does not describe retired reserve")
	if DisplayServer.get_name() != "headless":
		await RenderingServer.frame_post_draw
		root.get_texture().get_image().save_png("res://output/runtime_tests/survival_v2_game.png")
	scene.muted = true
	scene._stop_audio()
	await create_timer(0.06).timeout
	scene.queue_free()
	await process_frame
	print("SURVIVAL_V2_TESTS ", "PASS" if failures == 0 else "FAIL", " assertions=", assertions, " failures=", failures)
	quit(0 if failures == 0 else 1)
