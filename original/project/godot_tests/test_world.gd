extends SceneTree

const World = preload("res://scripts/world_state.gd")
var failures: int = 0

func check(value: bool, message: String) -> void:
	if not value:
		failures += 1
		push_error(message)

func _initialize() -> void:
	var a = World.new()
	var b = World.new()
	a.reset(2304)
	b.reset(2304)
	check(a.terrain == b.terrain, "Same seed must produce same terrain")
	check(a.resources == b.resources, "Same seed must produce same resource locations")
	var initial_tick: int = a.tick
	check(not a.apply_action({"verb": "teleport", "target_id": "0,0"}), "Unknown action must be rejected")
	check(a.tick == initial_tick, "Invalid actions must not mutate world time")
	check(a.is_walkable(a.player), "Spawn must be walkable")
	for action in a.allowed_actions():
		if action.verb == "move":
			var old: Vector2i = a.player
			check(a.apply_action(action), "Advertised move must execute")
			check(a.is_walkable(a.player), "Movement must not enter water")
			check(Vector2(old).distance_to(Vector2(a.player)) <= 1.01, "Movement must be one adjacent step")
			break
	a.inventory.berries = 1
	a.hunger = 30.0
	check(a.apply_action({"verb": "eat"}), "Eat should consume held food")
	check(a.inventory.berries == 0 and a.hunger > 40.0, "Eating must consume inventory and restore hunger")
	check(not a.apply_action({"verb": "eat"}), "Cannot eat without food")
	a.inventory.wood = 4
	a.inventory.stone = 2
	check(a.apply_action({"verb": "build_camp"}), "Camp should build with materials")
	check(a.camp.exists and a.inventory.wood == 0 and a.inventory.stone == 0, "Building must consume materials")
	check(not a.apply_action({"verb": "build_camp"}), "Cannot duplicate camp")
	a.health = 0.1
	a.hunger = 0.0
	a.thirst = 0.0
	check(a.apply_action({"verb": "wait"}), "Wait must execute")
	check(a.dead, "Starvation and thirst can kill")
	check(not a.apply_action({"verb": "wait"}), "Dead actor cannot act")
	var c = World.new()
	c.reset(2304)
	var drink_tile: Vector2i = Vector2i(-1, -1)
	for y in range(c.SIZE):
		for x in range(c.SIZE):
			var p := Vector2i(x, y)
			if c.is_walkable(p) and c.water_near(p):
				drink_tile = p
				break
		if drink_tile.x >= 0:
			break
	check(drink_tile.x >= 0, "World must contain reachable shores")
	c.player = drink_tile
	c.thirst = 20.0
	check(c.apply_action({"verb": "drink"}) and c.thirst > 40.0, "Drinking at shore must restore thirst")
	for test_seed in [2304, 701, 991]:
		var run = World.new()
		run.reset(test_seed)
		for _step in range(650):
			if run.dead:
				break
			var action: Dictionary = run.baseline_action()
			check(run.allowed_actions().has(action), "Baseline must only issue advertised legal actions")
			run.apply_action(action)
		check(bool(run.camp.exists), "Scripted comparator must build its first camp")
		check(not run.dead and run.day() >= 4, "Scripted comparator must demonstrate a viable survival loop across three days")
		print("BASELINE seed=", test_seed, " tick=", run.tick, " day=", run.day(), " health=", run.health, " camp=", run.camp.exists)
	print("WORLD_TESTS ", "PASS" if failures == 0 else "FAIL", " failures=", failures)
	quit(0 if failures == 0 else 1)
