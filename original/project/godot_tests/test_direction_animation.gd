extends SceneTree

const Motion = preload("res://scripts/actor_motion.gd")
const View = preload("res://scripts/world_view.gd")
const World = preload("res://scripts/world_state.gd")
const Physics = preload("res://scripts/physics_world.gd")
var failures: int = 0

func check(condition: bool, description: String) -> void:
	if not condition:
		failures += 1
		push_error(description)

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var motion = Motion.new()
	motion.reset(Vector2(100, 100))
	check(motion.facing == Motion.DOWN and motion.frame == 0, "Spawn faces down at rest")
	for step in [Vector2.LEFT, Vector2.RIGHT, Vector2.UP, Vector2.DOWN]:
		var expected: int = {Vector2.LEFT: Motion.LEFT, Vector2.RIGHT: Motion.RIGHT, Vector2.UP: Motion.UP, Vector2.DOWN: Motion.DOWN}[step]
		var start: Vector2 = motion.last_position
		motion.observe(start + step * 16.0, 0.07)
		check(motion.facing == expected, "Actual displacement selects correct facing")
		check(motion.moving and motion.frame != 0, "Walking has an actual stride frame")
		var phase_before: float = motion.travelled
		motion.observe(motion.last_position, 0.1)
		check(not motion.moving and motion.frame == 0, "Stationary wait or pause returns to idle")
		check(motion.facing == expected, "Idle preserves last heading")
		check(is_equal_approx(motion.travelled, phase_before), "Waiting does not advance stride")
		motion.reset(motion.last_position)
	motion.observe(motion.last_position + Vector2(200, 0), 0.016)
	check(not motion.moving and motion.frame == 0, "Teleport/reset is not a walking stride")
	var texture: Texture2D = load("res://assets/art/hero_directions_v1.png")
	check(texture != null and texture.get_width() == texture.get_height(), "Directional AI atlas loads as square 4x4 sheet")
	var view = View.new()
	view.view_size = Vector2(1920, 1080)
	view.camera = Vector2(1200, 900)
	var tile := Vector2i(30, 24)
	check(view.screen_to_tile(view.world_to_screen(Vector2(tile) * 52.0)) == tile, "Full-window screen/world projection roundtrip")
	view.free()
	await _physical_integration()
	print("DIRECTION_ANIMATION_TESTS ", "PASS" if failures == 0 else "FAIL", " failures=", failures)
	quit(0 if failures == 0 else 1)

func _physical_integration() -> void:
	var world = World.new()
	world.reset(2304)
	world.resources.clear()
	world.terrain.fill(World.GRASS)
	world.rebuild_spatial_index()
	var physics = Physics.new()
	root.add_child(physics)
	physics.sync_world(world, true)
	physics.set_physics_process(false)
	var view = View.new()
	view.world = world
	view.physics_world = physics
	view.view_size = root.get_visible_rect().size
	root.add_child(view)
	view.set_process(false)
	view.snap_camera()
	await physics_frame
	await physics_frame
	for direction in [Vector2i.LEFT, Vector2i.RIGHT, Vector2i.UP, Vector2i.DOWN]:
		var expected: int = {Vector2i.LEFT: Motion.LEFT, Vector2i.RIGHT: Motion.RIGHT, Vector2i.UP: Motion.UP, Vector2i.DOWN: Motion.DOWN}[direction]
		var destination: Vector2i = world.player + direction
		check(world.apply_action({"verb": "move", "target_id": world.tile_id(destination)}), "Real world accepts open-ground step")
		physics.sync_world(world)
		view._process(0.1)
		check(not view.actor_motion.moving, "Logical target alone must not animate walking")
		physics._physics_process(0.07)
		view._process(0.07)
		check(view.actor_motion.moving and view.actor_motion.facing == expected, "Native physical movement drives direction")
		check(view.actor_position.is_equal_approx(physics.actor_position_px()), "Rendered actor uses native position")
		if "--capture-directions" in OS.get_cmdline_user_args() and DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			var capture: Image = root.get_texture().get_image()
			DirAccess.make_dir_recursive_absolute("res://output/verification/m4")
			capture.save_png("res://output/verification/m4/hero_direction_%d.png" % expected)
			capture = null
		physics.frozen = true
		var stopped_at: Vector2 = physics.actor_position_px()
		physics._physics_process(0.1)
		view._process(0.1)
		check(not view.actor_motion.moving and view.actor_motion.frame == 0, "Paused unfinished step is idle, not walking in place")
		check(physics.actor_position_px().is_equal_approx(stopped_at), "Pause keeps real body still")
		physics.frozen = false
		physics._physics_process(1.0)
		view._process(0.2)
		view._process(0.1)
		check(not view.actor_motion.moving and view.actor_motion.facing == expected, "Arrival holds actual facing")
	check(physics.unexpected_collisions == 0, "Animation does not change collision semantics")
	view.queue_free()
	physics.queue_free()
	await process_frame
