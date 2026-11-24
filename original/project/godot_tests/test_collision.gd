extends SceneTree

const World = preload("res://scripts/world_state.gd")
var failures: int = 0

func check(condition: bool, description: String) -> void:
	if not condition:
		failures += 1
		push_error(description)

func _initialize() -> void:
	var world = World.new()
	world.reset(2304)
	for kind in ["tree", "rock", "berry"]:
		var resource: Dictionary = {}
		for candidate in world.resources:
			if str(candidate.kind) == kind:
				resource = candidate
				break
		check(not resource.is_empty(), "Fixture contains " + kind)
		if not resource.is_empty():
			var tile := Vector2i(int(resource.x), int(resource.y))
			check(not world.is_walkable(tile), kind + " footprint must block player movement")
	world.camp = {"exists": true, "x": 48, "y": 47}
	check(not world.is_walkable(Vector2i(48, 47)), "Camp footprint must block player movement")
	if world.has_method("rebuild_spatial_index"):
		_test_authority_and_lifecycle()
		_test_path_and_interaction()
		_test_camp_placement()
		_test_fixed_difficulty()
	print("COLLISION_LOGIC_TESTS ", "PASS" if failures == 0 else "FAIL", " failures=", failures)
	quit(0 if failures == 0 else 1)

func fixture():
	var world = World.new()
	world.reset(2304)
	world.resources.clear()
	world.terrain.fill(World.GRASS)
	world.camp = {"exists": false, "x": 48, "y": 48}
	world.player = Vector2i(48, 48)
	world.discovered.clear()
	world.rebuild_spatial_index()
	world._reveal()
	return world

func _test_authority_and_lifecycle() -> void:
	for kind in ["tree", "pine", "rock", "berry"]:
		var world = fixture()
		var obstacle := Vector2i(49, 48)
		world._add_resource(obstacle, kind)
		var action: Dictionary = {"verb": "move", "target_id": world.tile_id(obstacle)}
		check(not world.is_walkable(obstacle), kind + " occupies a logical footprint")
		check(not world.allowed_actions().has(action), kind + " blocked move must not be advertised")
		var before_tick: int = world.tick
		var before_energy: float = world.energy
		check(not world.apply_action(action), kind + " forged blocked move must be rejected")
		check(world.player == Vector2i(48, 48) and world.tick == before_tick and world.energy == before_energy,
			kind + " rejected collision must not advance time, position or energy")
		check(world.apply_action({"verb": "gather", "target_id": "r_49_48"}), kind + " must remain harvestable from beside it")
		check(not world.is_walkable(obstacle), kind + " remains solid while depleted (stump/rock/bush)")
		for i in range(95):
			world.apply_action({"verb": "wait"})
		check(not world.is_walkable(obstacle) and world.player != obstacle,
			kind + " regeneration must not create overlap with player")
	var passable = fixture()
	passable._add_resource(Vector2i(49, 48), "mushroom")
	check(passable.is_walkable(Vector2i(49, 48)), "Small mushroom decoration must be passable")
	check(passable.apply_action({"verb": "move", "target_id": "tile_49_48"}), "Player can walk through non-solid mushroom")

func _test_path_and_interaction() -> void:
	var world = fixture()
	world._add_resource(Vector2i(49, 48), "tree")
	world._add_resource(Vector2i(50, 48), "rock")
	var goal := Vector2i(51, 48)
	var reached: bool = false
	for i in range(20):
		if world.player == goal:
			reached = true
			break
		var previous: Vector2i = world.player
		var action: Dictionary = world.step_toward(goal)
		check(str(action.verb) == "move", "A route exists around the two obstacles")
		check(world.apply_action(action), "Route step must be an advertised executable action")
		check(world.is_walkable(world.player), "Route must never enter an obstacle")
		check(absi(world.player.x - previous.x) + absi(world.player.y - previous.y) == 1,
			"Route must use cardinal steps, not cut a diagonal corner")
	check(reached, "Route must reach the free destination rather than stall")
	world = fixture()
	var target := Vector2i(51, 48)
	world._add_resource(target, "tree")
	for i in range(12):
		if Vector2(world.player).distance_to(Vector2(target)) <= 1.5:
			break
		var action: Dictionary = world.step_toward(target)
		check(world.apply_action(action), "Approach to solid resource must be executable")
		check(world.player != target, "Resource approach must not enter target volume")
	check(Vector2(world.player).distance_to(Vector2(target)) <= 1.5, "Resource path must stop within interaction range")
	check(world.apply_action({"verb": "gather", "target_id": "r_51_48"}), "Resource should be harvestable after adjacent approach")
	world = fixture()
	for direction in World.DIRECTIONS:
		world._add_resource(world.player + direction, "rock")
	var stalled: Dictionary = world.step_toward(Vector2i(52, 48))
	check(stalled == {"verb": "wait"}, "Fully enclosed actor must not teleport through blockers")

func _test_camp_placement() -> void:
	var world = fixture()
	world.inventory.wood = 4
	world.inventory.stone = 2
	var actor_start: Vector2i = world.player
	check(world.apply_action({"verb": "build_camp"}), "Camp can be built on surrounding clear land")
	var camp_tile := Vector2i(int(world.camp.x), int(world.camp.y))
	check(camp_tile != actor_start, "Camp must not be placed around the standing player")
	check(not world.is_walkable(camp_tile), "Built camp must be a solid footprint")
	check(world.is_walkable(world.player), "Building must leave player on walkable ground")
	var exits: int = 0
	for action in world.allowed_actions():
		if str(action.verb) == "move":
			exits += 1
	check(exits > 0, "Building must retain a player exit")
	world.reset(2304)
	check(not world.camp.exists, "Reset removes previous camp state")
	check(world.is_walkable(world.player), "Reset respawns the player outside obstacles")
	world = fixture()
	world.inventory.wood = 4
	world.inventory.stone = 2
	for direction in [Vector2i.UP, Vector2i.LEFT, Vector2i.RIGHT]:
		world._add_resource(world.player + direction, "rock")
	check(not world.allowed_actions().has({"verb": "build_camp"}), "Building must not seal the only remaining exit")

func _test_fixed_difficulty() -> void:
	var teaching = World.new()
	var standard = World.new()
	teaching.reset(2304, "d0")
	standard.reset(2304, "d1")
	teaching.tick = 140
	standard.tick = 140
	teaching.apply_action({"verb": "wait"})
	standard.apply_action({"verb": "wait"})
	check(teaching.health == 100.0, "D0 must disable night damage")
	check(standard.health < 100.0, "D1 must retain exposed night damage")
	check(teaching.visible_threats().is_empty(), "D0 must disable dangerous enemies")
	check(is_equal_approx(82.0 - teaching.hunger, (82.0 - standard.hunger) * 0.5), "D0 fixed drain multiplier must be half D1")
