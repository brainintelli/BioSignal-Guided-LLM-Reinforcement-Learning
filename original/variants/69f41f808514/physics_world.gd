extends Node2D
## Native bodies mirror the deterministic world's shared foot-level geometry.

var world: MeadowWorld
var player_body: CharacterBody2D
var static_root: Node2D
var target_px: Vector2
var frozen: bool = false
var complete_paused_step: bool = false
var revision: int = -1
var movement_speed: float = 230.0
var unexpected_collisions: int = 0

func _ready() -> void:
	static_root = Node2D.new()
	static_root.name = "SolidFootprints"
	add_child(static_root)
	player_body = CharacterBody2D.new()
	player_body.name = "PlayerBody"
	player_body.collision_layer = 1
	player_body.collision_mask = 2
	player_body.motion_mode = CharacterBody2D.MOTION_MODE_FLOATING
	player_body.safe_margin = 0.01
	var foot := CollisionShape2D.new()
	foot.name = "FootCollision"
	var circle := CircleShape2D.new()
	circle.radius = MeadowWorld.PLAYER_RADIUS
	foot.shape = circle
	player_body.add_child(foot)
	add_child(player_body)
	if world != null:
		sync_world(world, true)

func sync_world(state: MeadowWorld, reset_actor: bool = false) -> void:
	world = state
	if player_body == null:
		return
	if revision != world.collision_revision:
		# Called between physics frames; freeing now prevents one-frame ghost solids.
		for body in static_root.get_children():
			body.free()
		for spec in world.collision_footprints():
			var solid := StaticBody2D.new()
			solid.collision_layer = 2
			solid.collision_mask = 1
			solid.position = spec.center_px
			solid.set_meta("kind", spec.kind)
			var collider := CollisionShape2D.new()
			if str(spec.shape) == "circle":
				var circle := CircleShape2D.new()
				circle.radius = float(spec.radius)
				collider.shape = circle
			else:
				var rectangle := RectangleShape2D.new()
				rectangle.size = spec.size
				collider.shape = rectangle
			solid.add_child(collider)
			static_root.add_child(solid)
		revision = world.collision_revision
	target_px = Vector2(world.player) * MeadowWorld.TILE_SIZE + MeadowWorld.FOOT_OFFSET
	if reset_actor:
		complete_paused_step = false
		player_body.position = target_px
		complete_paused_step = false
		player_body.velocity = Vector2.ZERO

func actor_position_px() -> Vector2:
	return player_body.position - MeadowWorld.FOOT_OFFSET if player_body != null else Vector2.ZERO

func is_moving() -> bool:
	return player_body != null and player_body.position.distance_squared_to(target_px) > 0.01

func can_move_to(tile: Vector2i) -> bool:
	var motion: Vector2 = Vector2(tile) * MeadowWorld.TILE_SIZE + MeadowWorld.FOOT_OFFSET - player_body.position
	return not player_body.test_move(player_body.transform, motion, null, 0.01)

func _physics_process(delta: float) -> void:
	if (frozen and not complete_paused_step) or not is_moving():
		return
	var displacement: Vector2 = target_px - player_body.position
	var motion: Vector2 = displacement.limit_length(movement_speed * delta)
	var collision: KinematicCollision2D = player_body.move_and_collide(motion)
	if collision != null:
		unexpected_collisions += 1
		# Native collision is authoritative for presentation; never snap through it.
		target_px = player_body.position
		complete_paused_step = false
		push_error("Native collision disagreed with the shared grid footprint; movement stopped")
	elif player_body.position.distance_squared_to(target_px) <= 0.01:
		player_body.position = target_px
		complete_paused_step = false
