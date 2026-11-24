extends Node2D
## Visual-only layer; the simulation remains deterministic and frame independent.

const TILE: float = 52.0
const ActorMotion = preload("res://scripts/actor_motion.gd")
# Measured opaque foot baselines in the original 1254 px AI sheet, per 313.5 px cell.
# Keeps all sixteen poses grounded without editing or resampling the source alpha.
const HERO_FOOT_BOTTOM: Array = [[295.0, 298.0, 298.0, 298.0], [287.5, 283.5, 286.5, 283.5], [282.0, 277.0, 282.0, 279.0], [284.5, 284.5, 283.5, 284.5]]
var world: MeadowWorld
var physics_world: Node2D
var debug_collision: bool = false
var view_size: Vector2 = Vector2(1080, 900)
var camera: Vector2 = Vector2(48, 48) * TILE
var actor_position: Vector2 = Vector2(48, 48) * TILE
var atlas: Texture2D
var hero_atlas: Texture2D
var actor_motion := ActorMotion.new()
var animation_time: float = 0.0
var hover_tile: Vector2i = Vector2i(-1, -1)
var marker: Vector2i = Vector2i(-1, -1)
var marker_age: float = 10.0
var cached_props: Array[Dictionary] = []
var cached_tile: Vector2i = Vector2i(-1, -1)
var cached_tick: int = -1
var cached_view_size := Vector2.ZERO
var draw_items: Array[Dictionary] = []
var water_styles: Array[StyleBoxFlat] = []

func _ready() -> void:
	if ResourceLoader.exists("res://assets/art/world_atlas.png"):
		atlas = load("res://assets/art/world_atlas.png") as Texture2D
	if ResourceLoader.exists("res://assets/art/hero_directions_v1.png"):
		hero_atlas = load("res://assets/art/hero_directions_v1.png") as Texture2D
	texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR
	for mask in range(16):
		var style := StyleBoxFlat.new()
		style.bg_color = Color("68cbd3")
		style.border_color = Color("b7e9cf")
		style.border_width_top = 3 if mask & 1 else 0
		style.border_width_right = 3 if mask & 2 else 0
		style.border_width_bottom = 3 if mask & 4 else 0
		style.border_width_left = 3 if mask & 8 else 0
		style.corner_radius_top_left = 20 if mask & 1 and mask & 8 else 0
		style.corner_radius_top_right = 20 if mask & 1 and mask & 2 else 0
		style.corner_radius_bottom_right = 20 if mask & 4 and mask & 2 else 0
		style.corner_radius_bottom_left = 20 if mask & 4 and mask & 8 else 0
		water_styles.append(style)

func _process(delta: float) -> void:
	if world == null:
		return
	animation_time += delta
	marker_age += delta
	var target: Vector2 = Vector2(world.player) * TILE
	actor_position = physics_world.actor_position_px() if physics_world != null else target
	actor_motion.observe(actor_position, delta)
	camera = camera.lerp(actor_position, 1.0 - exp(-5.5 * delta))
	if cached_tile != world.player or cached_tick != world.tick or cached_view_size != view_size:
		_refresh_props()
	queue_redraw()

func snap_camera() -> void:
	actor_position = Vector2(world.player) * TILE
	actor_motion.reset(actor_position)
	camera = actor_position
	cached_tile = Vector2i(-1, -1)
	_refresh_props()

func screen_to_tile(point: Vector2) -> Vector2i:
	var p: Vector2 = (point - view_size * Vector2(0.5, 0.53) + camera) / TILE
	return Vector2i(roundi(p.x), roundi(p.y))

func world_to_screen(p: Vector2) -> Vector2:
	return p - camera + view_size * Vector2(0.5, 0.53)

func _refresh_props() -> void:
	cached_tile = world.player
	cached_tick = world.tick
	cached_view_size = view_size
	cached_props.clear()
	var radius: Vector2i = _visible_radius()
	for r in world.resources:
		if abs(int(r.x) - world.player.x) <= radius.x + 2 and abs(int(r.y) - world.player.y) <= radius.y + 2:
			cached_props.append(r)

func _visible_radius() -> Vector2i:
	return Vector2i(ceili(view_size.x / (TILE * 2.0)) + 2, ceili(view_size.y / (TILE * 2.0)) + 2)

func _draw() -> void:
	if world == null:
		return
	draw_rect(Rect2(Vector2.ZERO, view_size), Color("a8d97a"))
	var center := Vector2i(roundi(camera.x / TILE), roundi(camera.y / TILE))
	var radius: Vector2i = _visible_radius()
	for y in range(center.y - radius.y, center.y + radius.y + 1):
		for x in range(center.x - radius.x, center.x + radius.x + 1):
			_draw_tile(Vector2i(x, y))
	# Gentle motes and drifting water highlights give life without changing game state.
	for i in range(14):
		var fx: float = fposmod(float(i * 137) + animation_time * (6.0 + i % 3), view_size.x)
		var fy: float = fposmod(float(i * 223) + sin(animation_time * 0.4 + float(i)) * 16.0, view_size.y)
		draw_circle(Vector2(fx, fy), 1.5, Color(1.0, 1.0, 0.85, 0.65))
	if hover_tile.x >= 0 and world.discovered.has(world.tile_id(hover_tile)):
		var hp: Vector2 = world_to_screen(Vector2(hover_tile) * TILE)
		draw_rect(Rect2(hp - Vector2.ONE * 24.0, Vector2.ONE * 48.0), Color(1, 1, 0.87, 0.17))
		draw_arc(hp, 22.0, 0.0, TAU, 28, Color(1.0, 1.0, 0.91, 0.7), 2.0, true)
	if marker_age < 1.2:
		var mp: Vector2 = world_to_screen(Vector2(marker) * TILE)
		draw_arc(mp, 12.0 + marker_age * 18.0, 0.0, TAU, 28, Color(1, 1, 1, 1.0 - marker_age / 1.2), 2.0, true)
	draw_items.clear()
	for r in cached_props:
		draw_items.append({"y": float(r.y) * TILE, "type": "resource", "data": r})
	if bool(world.camp.exists):
		draw_items.append({"y": float(world.camp.y) * TILE, "type": "camp"})
	for t in world.visible_threats():
		draw_items.append({"y": float(t.y) * TILE, "type": "threat", "data": t})
	if not world.is_night():
		for original_tile in [Vector2i(47, 43), Vector2i(53, 52)]:
			var rabbit_tile: Vector2i = world._layout_point(original_tile)
			if world.is_walkable(rabbit_tile):
				draw_items.append({"y": float(rabbit_tile.y) * TILE, "type": "rabbit", "tile": rabbit_tile})
	draw_items.append({"y": actor_position.y, "type": "actor"})
	draw_items.sort_custom(func(a: Dictionary, b: Dictionary) -> bool: return float(a.y) < float(b.y))
	for item in draw_items:
		match str(item.type):
			"resource":
				_draw_resource(item.data)
			"camp":
				_draw_camp()
			"threat":
				var t: Dictionary = item.data
				var p: Vector2 = world_to_screen(Vector2(float(t.x), float(t.y)) * TILE)
				_draw_asset(11, p + Vector2(0, sin(animation_time * 3.0) * 3.0), Vector2.ONE * 55.0)
			"actor":
				_draw_actor()
			"rabbit":
				var rabbit_tile: Vector2i = item.tile
				var rabbit_at: Vector2 = world_to_screen(Vector2(rabbit_tile) * TILE)
				var idle: float = maxf(0.0, sin(animation_time * 1.8 + rabbit_tile.x)) * 4.0
				_draw_asset(10, rabbit_at + Vector2(sin(animation_time * 0.3) * 8.0, -idle), Vector2(42, 42))
	# Soft uncovered map edge instead of a black shroud; discovered cells persist.
	for y in range(center.y - radius.y, center.y + radius.y + 1):
		for x in range(center.x - radius.x, center.x + radius.x + 1):
			var p := Vector2i(x, y)
			if not world.discovered.has(world.tile_id(p)):
				var sp: Vector2 = world_to_screen(Vector2(p) * TILE)
				draw_rect(Rect2(sp - Vector2.ONE * TILE * 0.5, Vector2.ONE * TILE), Color(0.89, 0.95, 0.85, 0.78))
	_draw_build_preview()
	if debug_collision:
		_draw_collision_overlay()
	if world.phase() == "dusk":
		draw_rect(Rect2(Vector2.ZERO, view_size), Color(1.0, 0.64, 0.38, 0.07))
	if world.is_night():
		draw_rect(Rect2(Vector2.ZERO, view_size), Color(0.16, 0.23, 0.45, 0.27))
		if bool(world.camp.exists):
			var cp: Vector2 = world_to_screen(Vector2(float(world.camp.x), float(world.camp.y)) * TILE)
			for i in range(6, 0, -1):
				draw_circle(cp, float(i) * 18.0, Color(1.0, 0.77, 0.33, 0.025))

func _draw_tile(p: Vector2i) -> void:
	var screen: Vector2 = world_to_screen(Vector2(p) * TILE)
	var rect := Rect2(screen - Vector2.ONE * TILE * 0.5, Vector2.ONE * (TILE + 0.5))
	var kind: int = world.terrain_at(p)
	var h: int = absi(p.x * 73856093 ^ p.y * 19349663 ^ world.world_seed)
	var variation: float = float(h % 9) * 0.0025
	var color := Color("a8d87c")
	match kind:
		MeadowWorld.WATER:
			color = Color("68cbd3")
		MeadowWorld.SAND:
			color = Color("e8db9c")
		MeadowWorld.FOREST:
			color = Color("8bc66e")
		MeadowWorld.FLOWERS:
			color = Color("b5d989")
	color = color.lightened(variation)
	draw_rect(rect, color)
	if kind == MeadowWorld.WATER:
		var mask: int = 0
		for i in range(4):
			if world.terrain_at(p + MeadowWorld.DIRECTIONS[i]) != MeadowWorld.WATER:
				mask |= 1 << i
		if mask != 0 and water_styles.size() == 16:
			draw_rect(rect, Color("e8db9c"))
			draw_style_box(water_styles[mask], rect)
		elif mask == 0:
			# Clear depth contrast makes the impassable lake readable at a glance.
			draw_rect(rect, Color("60c3d2").lightened(variation))
		var wave: float = sin(animation_time * 1.2 + float(h % 20))
		if h % 3 == 0:
			draw_line(screen + Vector2(-10, wave * 4), screen + Vector2(10, wave * 4), Color(0.79, 0.98, 0.96, 0.5), 2.0, true)
			if mask != 0:
				draw_line(screen + Vector2(-4, 10 + wave * 2), screen + Vector2(7, 10 + wave * 2), Color(0.92, 1.0, 0.87, 0.4), 1.4, true)
	elif h % 4 == 0:
		var offset := Vector2(float(h % 25) - 12.0, float((h / 25) % 25) - 12.0)
		var blade: Color = Color("78b965") if kind != MeadowWorld.SAND else Color("c9bc83")
		draw_line(screen + offset, screen + offset + Vector2(-3, -5), blade, 1.5, true)
		draw_line(screen + offset, screen + offset + Vector2(3, -6), blade, 1.5, true)
	if kind == MeadowWorld.FOREST and h % 3 == 0:
		# Ground dapple is visual-only: no extra entities, collision or model tokens.
		draw_circle(screen + Vector2(-9, 4), 13.0, Color(0.27, 0.50, 0.24, 0.06))
		draw_circle(screen + Vector2(10, -8), 8.0, Color(0.92, 0.98, 0.71, 0.10))
	var flower_patch: bool = kind == MeadowWorld.FLOWERS and h % 5 == 0
	if flower_patch or kind != MeadowWorld.WATER and kind != MeadowWorld.SAND and h % 17 == 0:
		_draw_asset(6, screen + Vector2(-5 if flower_patch else 0, 8), Vector2(30, 30), Color(1, 1, 1, 0.9))
		if flower_patch:
			_draw_asset(6, screen + Vector2(14, -11), Vector2(19, 19), Color(1.0, 0.93, 0.98, 0.88))
	elif kind == MeadowWorld.FOREST and h % 7 == 0:
		_draw_asset(15, screen, Vector2(35, 35))
	elif kind == MeadowWorld.SAND and h % 7 == 0:
		_draw_asset(7, screen, Vector2(42, 42))

func _draw_resource(r: Dictionary) -> void:
	var p: Vector2 = world_to_screen(Vector2(float(r.x), float(r.y)) * TILE)
	var available: bool = int(r.ready_at) <= world.tick
	var index: int = 3
	var size := Vector2(63, 63)
	match str(r.kind):
		"tree":
			index = 1 if available else 14
			size = Vector2(112, 112) if available else Vector2(48, 48)
		"pine":
			index = 2 if available else 14
			size = Vector2(104, 120) if available else Vector2(48, 48)
		"berry":
			index = 3
		"rock":
			index = 4
			size = Vector2(60, 56) if available else Vector2(36, 32)
		"mushroom":
			index = 5
			size = Vector2(48, 48) if available else Vector2(24, 24)
	draw_set_transform(p + Vector2(0, 8), 0.0, Vector2(1.0, 0.35))
	draw_circle(Vector2.ZERO, size.x * 0.25, Color(0.2, 0.35, 0.2, 0.15))
	draw_set_transform(Vector2.ZERO)
	var sway: float = sin(animation_time * 1.1 + float(r.x)) * 1.4 if available else 0.0
	var tint: Color = Color.WHITE if available else Color(0.85, 0.85, 0.8)
	# Keep the moving character visible behind canopies; collision stays unchanged.
	if available and str(r.kind) in ["tree", "pine"]:
		var resource_position := Vector2(float(r.x), float(r.y)) * TILE
		var separation: Vector2 = actor_position - resource_position
		if absf(separation.x) < size.x * 0.38 and separation.y < 0.0 and separation.y > -size.y * 0.85:
			tint.a = 0.48
	_draw_asset(index, p + MeadowWorld.FOOT_OFFSET + Vector2(sway, -size.y * 0.45), size, tint)

func _draw_actor() -> void:
	var p: Vector2 = world_to_screen(actor_position)
	var moving: bool = actor_motion.moving
	var stride: float = actor_motion.travelled / ActorMotion.FRAME_DISTANCE
	var hop: float = absf(sin(stride * PI * 0.5)) * 1.7 if moving else sin(animation_time * 2.0) * 0.6
	draw_set_transform(p + Vector2(0, 10), 0.0, Vector2(1.0, 0.4))
	draw_circle(Vector2.ZERO, 18.0, Color(0.14, 0.31, 0.23, 0.25))
	draw_set_transform(Vector2.ZERO)
	draw_arc(p + Vector2(0, 8), 23.0, 0, TAU, 32, Color(1, 1, 0.89, 0.6), 1.5, true)
	if hero_atlas != null:
		var cell: Vector2 = Vector2(hero_atlas.get_size()) / 4.0
		var source := Rect2(Vector2(float(actor_motion.frame), float(actor_motion.facing)) * cell, cell)
		var size := Vector2(86, 86)
		var baseline: float = float(HERO_FOOT_BOTTOM[actor_motion.facing][actor_motion.frame]) / 313.5
		var center: Vector2 = p + MeadowWorld.FOOT_OFFSET + Vector2(0, -(baseline - 0.5) * size.y - hop)
		draw_texture_rect_region(hero_atlas, Rect2(center - size * 0.5, size), source)
	else:
		_draw_asset(0, p + MeadowWorld.FOOT_OFFSET + Vector2(0, -78.0 * 0.43 - hop), Vector2(78, 78))
	# A readable nameplate is drawn in UI rather than baked into the generated asset.

func _draw_camp() -> void:
	var p: Vector2 = world_to_screen(Vector2(float(world.camp.x), float(world.camp.y)) * TILE)
	draw_circle(p, 58.0, Color(0.87, 0.83, 0.57, 0.38))
	_draw_asset(9, p + MeadowWorld.FOOT_OFFSET + Vector2(0, -105.0 * 0.43), Vector2(105, 105))
	_draw_asset(8, p + Vector2(-36, 17 + sin(animation_time * 8.0)), Vector2(42, 42))

func _draw_collision_overlay() -> void:
	var solid_color := Color(0.95, 0.32, 0.22, 0.9)
	for y in range(world.player.y - 10, world.player.y + 11):
		for x in range(world.player.x - 12, world.player.x + 13):
			var tile := Vector2i(x, y)
			if not world.discovered.has(world.tile_id(tile)):
				continue
			var spec: Dictionary = world.footprint_at(tile)
			if spec.is_empty():
				continue
			var center: Vector2 = world_to_screen(spec.center_px)
			if str(spec.shape) == "circle":
				draw_circle(center, float(spec.radius), Color(solid_color, 0.16))
				draw_arc(center, float(spec.radius), 0, TAU, 32, solid_color, 2.0, true)
			else:
				var dimensions: Vector2 = spec.size
				draw_rect(Rect2(center - dimensions * 0.5, dimensions), Color(solid_color, 0.10))
				draw_rect(Rect2(center - dimensions * 0.5, dimensions), solid_color, false, 1.5)
	var foot: Vector2 = world_to_screen(actor_position + MeadowWorld.FOOT_OFFSET)
	draw_circle(foot, MeadowWorld.PLAYER_RADIUS, Color(0.1, 0.7, 0.95, 0.25))
	draw_arc(foot, MeadowWorld.PLAYER_RADIUS, 0, TAU, 32, Color(0.15, 0.7, 1.0), 2.0, true)

func _draw_build_preview() -> void:
	if not bool(world.camp.exists) and int(world.inventory.wood) >= 4 and int(world.inventory.stone) >= 2:
		var site: Vector2i = world.camp_placement()
		if site.x >= 0:
			var center: Vector2 = world_to_screen(Vector2(site) * TILE + MeadowWorld.FOOT_OFFSET)
			draw_rect(Rect2(center - Vector2(22, 16), Vector2(44, 32)), Color(1.0, 0.8, 0.1), false, 2.0)

func _draw_asset(index: int, center: Vector2, size: Vector2, color: Color = Color.WHITE) -> void:
	if atlas != null:
		var cell: Vector2 = Vector2(atlas.get_size()) / 4.0
		var source := Rect2(Vector2(float(index % 4), float(index / 4)) * cell, cell)
		draw_texture_rect_region(atlas, Rect2(center - size * 0.5, size), source, color)
	else:
		# Temporary geometric fallback only when AI-generated assets are unavailable.
		var tint := Color("64a959")
		if index == 0:
			tint = Color("edb76b")
		elif index == 4:
			tint = Color("a7b4aa")
		elif index == 5:
			tint = Color("ec9285")
		elif index == 8:
			tint = Color("efb74f")
		elif index == 9:
			tint = Color("e4a770")
		draw_circle(center, size.x * 0.26, tint * color)
		if index == 0:
			draw_circle(center + Vector2(-7, -1), 2.5, Color("354b43"))
			draw_circle(center + Vector2(7, -1), 2.5, Color("354b43"))
