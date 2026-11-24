extends SceneTree
## Isolated scene test: instantiate production physics and perform real swept moves.

const World = preload("res://scripts/world_state.gd")
var failures: int = 0
var checks: int = 0

func check(condition: bool, description: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(description)

func _initialize() -> void:
	call_deferred("_run")

func fixture():
	var world = World.new()
	world.reset(2304)
	world.resources.clear()
	world.terrain.fill(World.GRASS)
	world.camp = {"exists": false, "x": 48, "y": 48}
	world.player = Vector2i(48, 48)
	world.rebuild_spatial_index()
	world.discovered.clear()
	world._reveal()
	return world

func _run() -> void:
	var physics_script = load("res://scripts/physics_world.gd")
	check(physics_script != null, "Production physics layer must load")
	if physics_script == null:
		quit(1)
		return
	for kind in ["tree", "pine", "rock", "berry", "camp", "water"]:
		var world = fixture()
		var obstacle := Vector2i(49, 48)
		if kind == "camp":
			world.camp = {"exists": true, "x": obstacle.x, "y": obstacle.y}
		elif kind == "water":
			world.terrain[obstacle.y * World.SIZE + obstacle.x] = World.WATER
		else:
			world._add_resource(obstacle, kind)
		world.rebuild_spatial_index()
		var physics = physics_script.new()
		root.add_child(physics)
		physics.sync_world(world, true)
		physics.set_physics_process(false)
		await physics_frame
		await physics_frame
		var body: CharacterBody2D = physics.player_body
		check(body != null, kind + " scene has the production player body")
		var origin: Vector2 = body.position
		var motion := Vector2(52.0 * 4.0, 0.0)
		var contact: KinematicCollision2D = body.move_and_collide(motion)
		check(contact != null, kind + " must physically stop a four-tile sweep")
		check(body.position.x > origin.x and body.position.x < origin.x + 52.0,
			kind + " player must stop before the obstacle center")
		check(absf(body.position.y - origin.y) < 0.1, kind + " collision must not unexpectedly teleport vertically")
		var collider_count: int = physics.static_root.get_child_count()
		check(collider_count > 0, kind + " fixture must register static colliders")
		physics.sync_world(world, true)
		await physics_frame
		check(physics.static_root.get_child_count() == collider_count, kind + " rebuild must not accumulate static bodies")
		check(physics.player_body.position.distance_to(origin) < 0.1, kind + " reset must restore actor outside obstacle")
		physics.queue_free()
		await process_frame
	var world = fixture()
	world._add_resource(Vector2i(49, 48), "tree")
	var physics = physics_script.new()
	root.add_child(physics)
	physics.sync_world(world, true)
	physics.set_physics_process(false)
	await physics_frame
	await physics_frame
	var body: CharacterBody2D = physics.player_body
	var origin: Vector2 = body.position
	for motion in [Vector2(0, -52), Vector2(104, 0), Vector2(0, 52)]:
		var contact: KinematicCollision2D = body.move_and_collide(motion)
		check(contact == null, "Open cardinal detour around trunk must not collide with canopy")
	check(body.position.distance_to(origin + Vector2(104, 0)) < 0.2, "Detour should arrive without clipping or drift")
	physics.queue_free()
	await process_frame
	await _test_authoritative_route(physics_script)
	await _test_map_bounds(physics_script)
	await _test_controller_input_gate()
	print("COLLISION_PHYSICS_TESTS ", "PASS" if failures == 0 else "FAIL", " checks=", checks, " failures=", failures)
	quit(0 if failures == 0 else 1)

func _test_authoritative_route(physics_script) -> void:
	var world = fixture()
	world._add_resource(Vector2i(49, 48), "tree")
	world._add_resource(Vector2i(50, 48), "rock")
	var physics = physics_script.new()
	root.add_child(physics)
	physics.sync_world(world, true)
	physics.set_physics_process(false)
	physics.movement_speed = 10000.0
	await physics_frame
	await physics_frame
	var destination := Vector2i(51, 48)
	for i in range(15):
		if world.player == destination:
			break
		var action: Dictionary = world.step_toward(destination)
		check(world.apply_action(action), "Production route must execute in authoritative world")
		physics.sync_world(world)
		physics._physics_process(1.0)
		check(not physics.is_moving(), "High-speed presentation must reach one route step")
		check(physics.actor_position_px().distance_to(Vector2(world.player) * World.TILE_SIZE) < 0.1,
			"Actual native actor position must agree with each authoritative path step")
		await physics_frame
	check(world.player == destination, "Native path follower must complete the detour")
	check(physics.unexpected_collisions == 0, "Valid detour must not trigger native/grid disagreement")
	physics.queue_free()
	await process_frame

func _test_map_bounds(physics_script) -> void:
	for pair in [[Vector2i(1, 48), Vector2(-208, 0)], [Vector2i(94, 48), Vector2(208, 0)],
		[Vector2i(48, 1), Vector2(0, -208)], [Vector2i(48, 94), Vector2(0, 208)]]:
		var world = fixture()
		world.player = pair[0]
		var physics = physics_script.new()
		root.add_child(physics)
		physics.sync_world(world, true)
		physics.set_physics_process(false)
		await physics_frame
		await physics_frame
		var contact: KinematicCollision2D = physics.player_body.move_and_collide(pair[1])
		check(contact != null, "Native body must collide with map boundary at " + str(pair[0]))
		var actual: Vector2 = physics.actor_position_px() / World.TILE_SIZE
		check(actual.x > 0.0 and actual.y > 0.0 and actual.x < World.SIZE - 1 and actual.y < World.SIZE - 1,
			"Native body must remain inside world bounds")
		physics.queue_free()
		await process_frame

func _test_controller_input_gate() -> void:
	var controller_script = load("res://scripts/main.gd")
	var controller = controller_script.new()
	root.add_child(controller)
	controller.set_process(false)
	controller._set_mode("human")
	controller.world = fixture()
	controller.world_view.world = controller.world
	controller.physics_world.revision = -1
	controller.physics_world.sync_world(controller.world, true)
	controller.physics_world.set_physics_process(false)
	await physics_frame
	await physics_frame
	var origin: Vector2 = controller.physics_world.actor_position_px()
	controller._execute({"verb": "move", "target_id": controller.world.tile_id(Vector2i(49, 48))}, "human", "QA first input")
	check(controller.world.tick == 1, "First controller input must consume one action")
	check(controller.physics_world.is_moving(), "Controller must wait for native motion completion")
	for i in range(20):
		controller._execute({"verb": "move", "target_id": controller.world.tile_id(Vector2i(50, 48))}, "human", "QA input burst")
	check(controller.world.tick == 1, "Twenty inputs during movement must not advance world again")
	check(controller.physics_world.actor_position_px().distance_to(origin) < 0.1, "Input burst must not teleport physical actor")
	controller.physics_world._physics_process(1.0)
	check(not controller.physics_world.is_moving(), "Physical motion must finish before next action")
	check(controller.physics_world.actor_position_px().distance_to(Vector2(49, 48) * World.TILE_SIZE) < 0.1,
		"Controller step must land at authoritative position")
	controller._execute({"verb": "move", "target_id": controller.world.tile_id(Vector2i(50, 48))}, "human", "QA next input")
	check(controller.world.tick == 2, "Next input may execute after physical completion")
	controller._restart()
	await physics_frame
	var count: int = controller.physics_world.static_root.get_child_count()
	controller._restart()
	await physics_frame
	check(controller.physics_world.static_root.get_child_count() == count, "Repeated real controller restart must not accumulate bodies")
	check(controller.physics_world.actor_position_px().distance_to(Vector2(controller.world.player) * World.TILE_SIZE) < 0.1,
		"Real controller restart must align native actor to new spawn")
	assert(controller.world.health == 100.0, "Restart health invariant")
	check(controller.world.health == 100.0 and not controller.physics_world.is_moving(), "Restart restores health and idle motion invariants")
	controller.queue_free()
	await process_frame
