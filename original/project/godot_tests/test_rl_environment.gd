extends SceneTree

const World = preload("res://scripts/world_state.gd")
const Session = preload("res://scripts/rl_environment_session.gd")
var failures: int = 0
var assertions: int = 0

func check(condition: bool, message: String) -> void:
	assertions += 1
	if not condition:
		failures += 1
		push_error(message)

func _initialize() -> void:
	var session = Session.new()
	check(not session.handle({"id": 1, "op": "observe"}).ok, "Observation requires explicit reset")
	check(not session.handle({"op": "reset", "seed": 1.5}).ok, "Fractional seeds rejected")
	check(not session.handle({"op": "reset", "profile": "tutorial"}).ok, "No task curriculum accepted")
	var response: Dictionary = session.handle({"id": 7, "op": "reset", "seed": 2304})
	check(response.ok and response.id == 7 and response.protocol == "meadow_rl_v1", "Correlated reset response")
	check(response.observation.profile == "d1" and not response.observation.has("task"), "Default D1 without task")
	var before: Dictionary = session.snapshot()
	check(not session.handle({"op": "step", "action": {"verb": "move", "target_id": "tile_0_0"}}).ok, "Illegal teleport rejected")
	check(session.snapshot() == before, "Rejected action must not mutate the state")
	check(not session.handle({"op": "step", "action": {"verb": "wait", "extra": true}}).ok, "Extra action keys rejected")
	response = session.handle({"op": "step", "action": {"verb": "rest"}})
	check(response.ok and response.transition.elapsed_ticks == 3 and response.transition.trace.size() == 3, "Multi-tick actions expose exact trace")
	check(response.transition.before.tick == 0 and response.transition.after.tick == 3, "Transition boundaries")
	check(not response.has("reward"), "Bridge supplies facts without prescribing reward")
	var world: MeadowWorld = session.world
	world.health = 50.0
	world.hunger = 0.0
	world.thirst = 0.0
	world.camp = {"exists": true, "x": world.player.x + 1, "y": world.player.y}
	response = session.handle({"op": "step", "action": {"verb": "rest"}})
	var total_damage: float = 0.0
	for row in response.transition.trace:
		total_damage += float(row.damage)
	check(is_equal_approx(total_damage, 9.0), "Raw damage not hidden by pre-tick camp healing")
	check(is_equal_approx(float(response.transition.after.health), 45.0), "Rest healing and damage both retained")
	world.reset(2304)
	var observed: Dictionary = world.observation()
	check(int(observed.local_visits[2][2]) == 1, "Spawn counted as one actual tile entry")
	world.apply_action({"verb": "wait"})
	world.apply_action({"verb": "rest"})
	check(int(world.local_visit_counts()[2][2]) == 1, "Stationary actions do not inflate entry counts")
	var origin: Vector2i = world.player
	for action in world.allowed_actions():
		if str(action.verb) == "move":
			world.apply_action(action)
			check(int(world.local_visit_counts()[2][2]) == 1, "First actual move records one entry")
			check(world.apply_action({"verb": "move", "target_id": world.tile_id(origin)}), "Return move is legal")
			check(int(world.local_visit_counts()[2][2]) == 2, "Returning to a tile records a revisit")
			break
	world.reset(2304)
	check(world.entered_counts.size() == 1 and int(world.local_visit_counts()[2][2]) == 1, "Reset clears episode entry memory")
	check(observed.local_map.rows.size() == 5, "Bounded local map height")
	for row in observed.local_map.rows:
		check(str(row).length() == 5, "Bounded local map width")
	var unknown: Vector2i = world.player + Vector2i(1, 0)
	world.discovered.erase(world.tile_id(unknown))
	world.entered_counts[world.tile_id(unknown)] = 8
	var known_map: Dictionary = world.local_observation_map()
	check(str(known_map.rows[2]).substr(3, 1) == "?", "Unknown tile content never leaks into local map")
	check(int(world.local_visit_counts()[2][3]) == 0, "Unknown tile does not expose synthetic visit history")
	world.discovered[world.tile_id(unknown)] = true
	for landmark in world.observation().nearby:
		var route: Array[Vector2i] = world.path_to(Vector2i(int(landmark.x), int(landmark.y)))
		check(int(landmark.path_steps) == route.size() - 1, "Path facts share authoritative collision route")
	for symmetry in range(8):
		var plain = World.new()
		var varied = World.new()
		var seed_value: int = 2304 + symmetry
		plain.reset(seed_value, "d1")
		varied.reset(seed_value, "open1")
		check(varied.layout_symmetry == symmetry, "Seed determines reproducible layout symmetry")
		check(varied.player == varied._layout_point(plain.player), "Spawn transformed with topology")
		check(varied.is_walkable(varied.player), "Varied layout spawn remains legal")
		check(varied.entered_counts.size() == 1 and int(varied.local_visit_counts()[2][2]) == 1, "Visit memory starts at transformed spawn")
		check(varied.resources.size() == plain.resources.size(), "No extra resource complexity")
		for resource in plain.resources:
			var point: Vector2i = varied._layout_point(Vector2i(int(resource.x), int(resource.y)))
			check(varied.resource_at(point).get("kind", "") == resource.kind, "Resource transform preserves types and spatial index")
		for y in range(World.SIZE):
			for x in range(World.SIZE):
				var point := Vector2i(x, y)
				check(plain.terrain_at(point) == varied.terrain_at(varied._layout_point(point)), "Terrain symmetry is an exact bijection")
		plain.tick = 145
		varied.tick = 145
		var plain_threats: Array[Dictionary] = plain.visible_threats()
		var varied_threats: Array[Dictionary] = varied.visible_threats()
		check(plain_threats.size() == varied_threats.size(), "Threat visibility rotates with spawn")
		for i in range(plain_threats.size()):
			var target: Vector2i = varied._layout_point(Vector2i(int(plain_threats[i].x), int(plain_threats[i].y)))
			check(target == Vector2i(int(varied_threats[i].x), int(varied_threats[i].y)), "Threat positions transformed consistently")
		plain.apply_action({"verb": "wait"})
		varied.apply_action({"verb": "wait"})
		check(is_equal_approx(plain.health, varied.health) and is_equal_approx(plain.thirst, varied.thirst), "Open1 uses unchanged D1 damage and drain")
	world.health = 0.1
	world.thirst = 0.0
	response = session.handle({"op": "step", "action": {"verb": "wait"}})
	check(response.terminated and response.observation.allowed_actions.is_empty(), "Death terminates and masks actions")
	check(not session.handle({"op": "step", "action": {"verb": "wait"}}).ok, "Dead world rejects further actions")
	check(session.handle({"op": "close"}).closed, "Explicit close acknowledgment")
	print("RL_ENVIRONMENT_TESTS ", "PASS" if failures == 0 else "FAIL", " assertions=", assertions, " failures=", failures)
	quit(0 if failures == 0 else 1)
