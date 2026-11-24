class_name MeadowWorld
extends RefCounted
## Deterministic simulation. Rendering and HTTP latency never advance this clock.

const TILE_SIZE: float = 52.0
const PLAYER_RADIUS: float = 10.0
const FOOT_OFFSET: Vector2 = Vector2(0, 10)
const WORLD_VERSION: int = 2
const SIZE: int = 96
const DAY_STEPS: int = 180
const VIEW_RADIUS: int = 7
const DIRECTIONS: Array[Vector2i] = [Vector2i.UP, Vector2i.RIGHT, Vector2i.DOWN, Vector2i.LEFT]
const GRASS: int = 0
const WATER: int = 1
const SAND: int = 2
const FOREST: int = 3
const FLOWERS: int = 4

var terrain: PackedInt32Array = PackedInt32Array()
var resources: Array[Dictionary] = []
var discovered: Dictionary = {}
var entered_counts: Dictionary = {}
var player: Vector2i = Vector2i(48, 48)
var health: float = 100.0
var hunger: float = 82.0
var thirst: float = 78.0
var energy: float = 92.0
var inventory: Dictionary = {"wood": 0, "stone": 0, "berries": 2}
var camp: Dictionary = {"exists": false, "x": 48, "y": 48}
var tick: int = 0
var world_seed: int = 2304
var profile: String = "d1"
var layout_symmetry: int = 0
var collision_revision: int = 0
var resource_index: Dictionary = {}
var recent_actions: Array[Dictionary] = []
var last_step_trace: Array[Dictionary] = []
var _route_key: String = ""
var _route_parents: Dictionary = {}
var _route_depths: Dictionary = {}
var dead: bool = false
var last_event: String = "清晨的草地醒了，旅程由你决定。"
var last_verb: String = "wait"
var gathered: int = 0
var rng: RandomNumberGenerator = RandomNumberGenerator.new()

func reset(new_seed: int = 2304, difficulty: String = "") -> void:
	if not difficulty.is_empty():
		profile = difficulty if difficulty in ["d0", "d1", "open1"] else "d1"
	collision_revision += 1
	recent_actions.clear()
	last_step_trace.clear()
	resource_index.clear()
	world_seed = new_seed
	layout_symmetry = posmod(new_seed, 8) if profile == "open1" else 0
	rng.seed = new_seed
	tick = 0
	player = Vector2i(48, 48)
	health = 100.0
	hunger = 82.0
	thirst = 78.0
	energy = 92.0
	inventory = {"wood": 0, "stone": 0, "berries": 2}
	camp = {"exists": false, "x": 48, "y": 48}
	dead = false
	resources.clear()
	discovered.clear()
	entered_counts.clear()
	gathered = 0
	terrain.resize(SIZE * SIZE)
	var noise := FastNoiseLite.new()
	noise.seed = new_seed
	noise.frequency = 0.055
	noise.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	for y in range(SIZE):
		for x in range(SIZE):
			var v: float = noise.get_noise_2d(float(x), float(y))
			var kind: int = GRASS
			if v < -0.39:
				kind = WATER
			elif v < -0.26:
				kind = SAND
			elif v > 0.27:
				kind = FOREST
			elif v > 0.08:
				kind = FLOWERS
			terrain[y * SIZE + x] = kind
	# A guaranteed starting meadow and shore avoid unwinnable initial seeds.
	for y in range(40, 57):
		for x in range(37, 60):
			terrain[y * SIZE + x] = FOREST if x > 54 else GRASS
	for y in range(43, 55):
		for x in range(37, 47):
			var lake: float = pow((float(x) - 41.7) / 4.8, 2.0) + pow((float(y) - 48.4) / 5.0, 2.0)
			if lake < 1.0:
				terrain[y * SIZE + x] = WATER
			elif lake < 1.5:
				terrain[y * SIZE + x] = SAND
	for i in range(920):
		var p := Vector2i(rng.randi_range(2, SIZE - 3), rng.randi_range(2, SIZE - 3))
		if not is_walkable(p) or p.distance_to(player) < 2.0 or not resource_at(p).is_empty():
			continue
		var options: Array[String] = ["tree", "pine", "berry", "rock", "mushroom"]
		_add_resource(p, options[rng.randi_range(0, 4)])
	for spec in [
		[Vector2i(49, 46), "berry"], [Vector2i(51, 49), "tree"],
		[Vector2i(50, 51), "rock"], [Vector2i(47, 45), "berry"],
		[Vector2i(53, 46), "pine"], [Vector2i(52, 43), "tree"],
		[Vector2i(48, 53), "mushroom"], [Vector2i(55, 50), "pine"],
		[Vector2i(53, 53), "berry"], [Vector2i(46, 42), "rock"]
	]:
		var spot: Vector2i = spec[0]
		_remove_resource_at(spot)
		_add_resource(spot, str(spec[1]))
	if layout_symmetry != 0:
		_transform_start_layout()
	entered_counts[tile_id(player)] = 1
	_reveal()
	last_event = "清晨的草地醒了，旅程由你决定。"
	last_verb = "wait"

func _layout_point(point: Vector2i) -> Vector2i:
	# Eight square symmetries preserve distances, connectivity and collision shapes.
	var result: Vector2i = point
	if layout_symmetry >= 4:
		result.x = SIZE - 1 - result.x
	for _rotation in range(layout_symmetry % 4):
		result = Vector2i(SIZE - 1 - result.y, result.x)
	return result

func _transform_start_layout() -> void:
	var transformed: PackedInt32Array = terrain.duplicate()
	for y in range(SIZE):
		for x in range(SIZE):
			var point: Vector2i = _layout_point(Vector2i(x, y))
			transformed[point.y * SIZE + point.x] = terrain[y * SIZE + x]
	terrain = transformed
	for resource in resources:
		var point: Vector2i = _layout_point(Vector2i(int(resource.x), int(resource.y)))
		resource.x = point.x
		resource.y = point.y
		resource.id = "r_%d_%d" % [point.x, point.y]
	player = _layout_point(player)
	camp.x = player.x
	camp.y = player.y
	rebuild_spatial_index()

func _add_resource(p: Vector2i, kind: String) -> void:
	var resource: Dictionary = {"id": "r_%d_%d" % [p.x, p.y], "kind": kind, "x": p.x, "y": p.y, "ready_at": 0}
	resources.append(resource)
	resource_index[p] = resource
	collision_revision += 1

func _remove_resource_at(p: Vector2i) -> void:
	for i in range(resources.size() - 1, -1, -1):
		if int(resources[i].x) == p.x and int(resources[i].y) == p.y:
			resources.remove_at(i)
	resource_index.erase(p)
	collision_revision += 1

func terrain_at(p: Vector2i) -> int:
	if p.x < 0 or p.y < 0 or p.x >= SIZE or p.y >= SIZE:
		return WATER
	return terrain[p.y * SIZE + p.x]

func is_walkable(p: Vector2i) -> bool:
	return p.x > 0 and p.y > 0 and p.x < SIZE - 1 and p.y < SIZE - 1 and footprint_at(p).is_empty()

func rebuild_spatial_index() -> void:
	resource_index.clear()
	for resource in resources:
		resource_index[Vector2i(int(resource.x), int(resource.y))] = resource
	collision_revision += 1

func footprint_at(p: Vector2i) -> Dictionary:
	# Native shapes and deterministic walkability consume the same foot geometry.
	if p.x <= 0 or p.y <= 0 or p.x >= SIZE - 1 or p.y >= SIZE - 1:
		return {"shape": "rect", "center_px": Vector2(p) * TILE_SIZE, "size": Vector2.ONE * TILE_SIZE, "kind": "boundary", "tile": p}
	if terrain_at(p) == WATER:
		return {"shape": "rect", "center_px": Vector2(p) * TILE_SIZE, "size": Vector2.ONE * TILE_SIZE, "kind": "water", "tile": p}
	if bool(camp.exists) and p == Vector2i(int(camp.x), int(camp.y)):
		return {"shape": "rect", "center_px": Vector2(p) * TILE_SIZE + FOOT_OFFSET, "size": Vector2(44, 32), "kind": "camp", "tile": p}
	var resource: Dictionary = resource_at(p)
	if resource.is_empty():
		return {}
	var kind: String = str(resource.kind)
	if kind == "rock":
		return {"shape": "rect", "center_px": Vector2(p) * TILE_SIZE + FOOT_OFFSET, "size": Vector2(34, 26), "kind": kind, "tile": p}
	if kind in ["tree", "pine", "berry"]:
		return {"shape": "circle", "center_px": Vector2(p) * TILE_SIZE + FOOT_OFFSET, "radius": 16.0 if kind != "berry" else 14.0, "kind": kind, "tile": p}
	return {}

func collision_footprints() -> Array[Dictionary]:
	var result: Array[Dictionary] = []
	# The outermost tile ring is solid in both native physics and grid movement.
	var span: float = SIZE * TILE_SIZE
	for point in [Vector2(0, (SIZE - 1) * 0.5), Vector2(SIZE - 1, (SIZE - 1) * 0.5)]:
		result.append({"shape": "rect", "center_px": point * TILE_SIZE, "size": Vector2(TILE_SIZE, span), "kind": "boundary"})
	for point in [Vector2((SIZE - 1) * 0.5, 0), Vector2((SIZE - 1) * 0.5, SIZE - 1)]:
		result.append({"shape": "rect", "center_px": point * TILE_SIZE, "size": Vector2(span, TILE_SIZE), "kind": "boundary"})
	# Merge horizontal water runs: exact occupied area, fewer physics objects.
	for y in range(SIZE):
		var x: int = 0
		while x < SIZE:
			if terrain_at(Vector2i(x, y)) != WATER:
				x += 1
				continue
			var start: int = x
			while x < SIZE and terrain_at(Vector2i(x, y)) == WATER:
				x += 1
			result.append({"shape": "rect", "center_px": Vector2((float(start + x - 1)) * 0.5, y) * TILE_SIZE, "size": Vector2(float(x - start) * TILE_SIZE, TILE_SIZE), "kind": "water"})
	for resource in resources:
		var shape: Dictionary = footprint_at(Vector2i(int(resource.x), int(resource.y)))
		if not shape.is_empty() and str(shape.kind) != "water":
			result.append(shape)
	if bool(camp.exists):
		result.append(footprint_at(Vector2i(int(camp.x), int(camp.y))))
	return result

func can_interact(target: Vector2i) -> bool:
	# Cardinal reach avoids gathering diagonally through an intervening solid.
	return absi(target.x - player.x) + absi(target.y - player.y) <= 1

func camp_placement() -> Vector2i:
	var exits: Array[Vector2i] = []
	for direction in DIRECTIONS:
		if is_walkable(player + direction):
			exits.append(player + direction)
	# Preserve at least one exit for the builder.
	if exits.size() < 2:
		return Vector2i(-1, -1)
	for candidate in exits:
		if resource_at(candidate).is_empty():
			return candidate
	return Vector2i(-1, -1)

func water_near(p: Vector2i) -> bool:
	for d in DIRECTIONS:
		if terrain_at(p + d) == WATER:
			return true
	return false

func resource_at(p: Vector2i) -> Dictionary:
	if resource_index.size() != resources.size():
		rebuild_spatial_index()
	return resource_index.get(p, {})

func resource_by_id(id: String) -> Dictionary:
	for r in resources:
		if str(r.id) == id:
			return r
	return {}

func day() -> int:
	return tick / DAY_STEPS + 1

func phase() -> String:
	var t: int = tick % DAY_STEPS
	if t < 20:
		return "dawn"
	if t < 115:
		return "day"
	if t < 140:
		return "dusk"
	return "night"

func is_night() -> bool:
	return phase() == "night"

func camp_near() -> bool:
	return bool(camp.exists) and Vector2(player).distance_to(Vector2(float(camp.x), float(camp.y))) <= 2.5

func visible_resources() -> Array[Dictionary]:
	var found: Array[Dictionary] = []
	for r in resources:
		var dist: float = Vector2(player).distance_to(Vector2(float(r.x), float(r.y)))
		if dist <= VIEW_RADIUS:
			found.append({"id": r.id, "kind": r.kind, "x": r.x, "y": r.y, "distance": snappedf(dist, 0.1), "available": int(r.ready_at) <= tick, "ready_in": maxi(0, int(r.ready_at) - tick)})
	found.sort_custom(func(a: Dictionary, b: Dictionary) -> bool: return float(a.distance) < float(b.distance))
	var compact: Array[Dictionary] = []
	var counts: Dictionary = {}
	for item in found:
		var kind: String = str(item.kind)
		if int(counts.get(kind, 0)) < 2:
			compact.append(item)
			counts[kind] = int(counts.get(kind, 0)) + 1
	return compact

func visible_threats() -> Array[Dictionary]:
	var threats: Array[Dictionary] = []
	if profile == "d0" or not is_night():
		return threats
	for i in range(3):
		var p: Vector2i = _layout_point(Vector2i(51 + i * 3, 47 + i * 2))
		var dist: float = Vector2(p).distance_to(Vector2(player))
		if dist <= VIEW_RADIUS:
			threats.append({"id": "slime_%d" % i, "x": p.x, "y": p.y, "distance": snappedf(dist, 0.1)})
	return threats

func tile_id(p: Vector2i) -> String:
	return "tile_%d_%d" % [p.x, p.y]

func tile_from_id(id: String) -> Vector2i:
	var parts: PackedStringArray = id.split("_")
	if parts.size() != 3:
		return Vector2i(-1, -1)
	return Vector2i(int(parts[1]), int(parts[2]))

func allowed_actions() -> Array[Dictionary]:
	var result: Array[Dictionary] = []
	if dead:
		return result
	for d in DIRECTIONS:
		var p: Vector2i = player + d
		if is_walkable(p):
			result.append({"verb": "move", "target_id": tile_id(p)})
	for r in visible_resources():
		if can_interact(Vector2i(int(r.x), int(r.y))) and bool(r.available):
			result.append({"verb": "gather", "target_id": str(r.id)})
	if int(inventory.berries) > 0:
		result.append({"verb": "eat"})
	if water_near(player):
		result.append({"verb": "drink"})
	if not bool(camp.exists) and int(inventory.wood) >= 4 and int(inventory.stone) >= 2 and camp_placement().x >= 0:
		result.append({"verb": "build_camp"})
	for landmark in navigation_landmarks():
		var destination := Vector2i(int(landmark.x), int(landmark.y))
		if not can_interact(destination) or str(landmark.kind) == "water" and destination != player:
			var step: Dictionary = step_toward(destination)
			if str(step.verb) == "move":
				result.append({"verb": "navigate", "target_id": str(landmark.id)})
	result.append({"verb": "rest"})
	result.append({"verb": "wait"})
	return result

func navigation_landmarks() -> Array[Dictionary]:
	var nearby: Array[Dictionary] = visible_resources()
	var shores: Array[Dictionary] = []
	for y in range(player.y - VIEW_RADIUS, player.y + VIEW_RADIUS + 1):
		for x in range(player.x - VIEW_RADIUS, player.x + VIEW_RADIUS + 1):
			var p := Vector2i(x, y)
			var distance: float = Vector2(p).distance_to(Vector2(player))
			if distance <= VIEW_RADIUS and is_walkable(p) and water_near(p):
				shores.append({"id": tile_id(p), "kind": "water", "x": x, "y": y, "distance": snappedf(distance, 0.1), "available": true})
	shores.sort_custom(func(a: Dictionary, b: Dictionary) -> bool: return float(a.distance) < float(b.distance))
	# One nearest reachable shore represents the local water supply; no task ranking.
	for shore in shores:
		var position := Vector2i(int(shore.x), int(shore.y))
		if position == player or str(step_toward(position).verb) == "move":
			nearby.append(shore)
			break
	if bool(camp.exists):
		nearby.append({"id": "camp", "kind": "camp", "x": camp.x, "y": camp.y,
			"distance": snappedf(Vector2(player).distance_to(Vector2(float(camp.x), float(camp.y))), 0.1), "available": true})
	return nearby

func observation() -> Dictionary:
	var landmarks: Array[Dictionary] = navigation_landmarks()
	for landmark in landmarks:
		var route: Array[Vector2i] = path_to(Vector2i(int(landmark.x), int(landmark.y)))
		landmark["path_steps"] = route.size() - 1 if not route.is_empty() else -1
	return {"schema_version": 1, "world_version": WORLD_VERSION, "profile": profile,
		"tick": tick, "day": day(), "phase": phase(),
		"player": {"health": snappedf(health, 0.1), "hunger": snappedf(hunger, 0.1), "thirst": snappedf(thirst, 0.1), "energy": snappedf(energy, 0.1), "x": player.x, "y": player.y},
		"inventory": inventory.duplicate(), "nearby": landmarks, "threats": visible_threats(), "local_map": local_observation_map(), "local_visits": local_visit_counts(),
		"camp": camp.duplicate(), "allowed_actions": allowed_actions(), "visited_count": discovered.size(),
		"recent_actions": recent_actions.duplicate(true), "navigation_assistance": "selected_target_one_collision_safe_step"}

func local_observation_map() -> Dictionary:
	# Relative, bounded facts only. Never inspect an undiscovered cell's contents.
	var rows: Array[String] = []
	for dy in range(-2, 3):
		var row: String = ""
		for dx in range(-2, 3):
			var point: Vector2i = player + Vector2i(dx, dy)
			if not discovered.has(tile_id(point)):
				row += "?"
			elif terrain_at(point) == WATER:
				row += "~"
			elif not is_walkable(point):
				row += "#"
			else:
				row += [".", "~", "s", "f", "*"][terrain_at(point)]
		rows.append(row)
	return {"radius": 2, "rows": rows, "legend": {"?": "unknown", "~": "water_blocked", "#": "solid", ".": "grass", "s": "sand", "f": "forest", "*": "flowers"}, "center": "player", "north": "row0"}

func local_visit_counts() -> Array[Array]:
	# Actual tile entries, not visibility or time spent stationary; aligned to local_map.
	var rows: Array[Array] = []
	for dy in range(-2, 3):
		var row: Array[int] = []
		for dx in range(-2, 3):
			var id: String = tile_id(player + Vector2i(dx, dy))
			row.append(int(entered_counts.get(id, 0)) if discovered.has(id) else 0)
		rows.append(row)
	return rows

func apply_action(action: Dictionary) -> bool:
	last_step_trace.clear()
	if dead or not allowed_actions().has(action):
		return false
	var before: Dictionary = {"health": health, "hunger": hunger, "thirst": thirst, "energy": energy}
	var previous_position: Vector2i = player
	var verb: String = str(action.verb)
	last_verb = verb
	var cost: int = 1
	match verb:
		"move", "navigate":
			if verb == "navigate":
				for landmark in navigation_landmarks():
					if str(landmark.id) == str(action.target_id):
						var next: Dictionary = step_toward(Vector2i(int(landmark.x), int(landmark.y)))
						player = tile_from_id(str(next.target_id))
						break
			else:
				player = tile_from_id(str(action.target_id))
			energy = maxf(0.0, energy - 0.35)
			last_event = "走到 (%d, %d)，沿途发现新的风景。" % [player.x, player.y]
		"gather":
			var resource: Dictionary = resource_by_id(str(action.target_id))
			var kind: String = str(resource.kind)
			if kind == "tree" or kind == "pine":
				inventory.wood = int(inventory.wood) + 2
				last_event = "收集木材 +2。林木将在一段时间后恢复。"
			elif kind == "rock":
				inventory.stone = int(inventory.stone) + 2
				last_event = "拾取石块 +2，可以用于营地建设。"
			else:
				inventory.berries = int(inventory.berries) + 3
				last_event = "收集可食浆果 +3，收进背包。"
			resource.ready_at = tick + 90
			energy = maxf(0.0, energy - 2.0)
			gathered += 1
			cost = 2
		"eat":
			inventory.berries = int(inventory.berries) - 1
			hunger = minf(100.0, hunger + 18.0)
			last_event = "吃下一份野果，饱食 +18。"
		"drink":
			thirst = minf(100.0, thirst + 30.0)
			last_event = "在清澈的岸边饮水，水分 +30。"
		"build_camp":
			inventory.wood = int(inventory.wood) - 4
			inventory.stone = int(inventory.stone) - 2
			var site: Vector2i = camp_placement()
			camp = {"exists": true, "x": site.x, "y": site.y}
			collision_revision += 1
			last_event = "营地建好了！在附近休息更安全，也能恢复生命。"
			cost = 4
		"rest":
			energy = minf(100.0, energy + 14.0)
			if camp_near():
				health = minf(100.0, health + 4.0)
			last_event = "在营地休息，恢复体力与生命。" if camp_near() else "坐下来稍作休息，恢复体力。"
			cost = 3
		"wait":
			last_event = "观察周围，等待一个世界步。"
	if player != previous_position:
		var entered_id: String = tile_id(player)
		entered_counts[entered_id] = int(entered_counts.get(entered_id, 0)) + 1
	_advance(cost)
	_reveal()
	recent_actions.append({"action": action.duplicate(), "position": {"x": player.x, "y": player.y},
		"delta": {"health": snappedf(health - float(before.health), 0.1), "hunger": snappedf(hunger - float(before.hunger), 0.1),
			"thirst": snappedf(thirst - float(before.thirst), 0.1), "energy": snappedf(energy - float(before.energy), 0.1)}, "tick": tick})
	if recent_actions.size() > 4:
		recent_actions.pop_front()
	return true

func _advance(steps: int) -> void:
	for _i in range(steps):
		var health_before_tick: float = health
		tick += 1
		var drain: float = 0.5 if profile == "d0" else 1.0
		hunger = maxf(0.0, hunger - 0.35 * drain)
		thirst = maxf(0.0, thirst - 0.45 * drain)
		if hunger <= 0.0:
			health -= 1.2
		if thirst <= 0.0:
			health -= 1.8
		if energy <= 0.0:
			health -= 0.6
		if profile != "d0" and is_night() and not camp_near():
			health -= 0.3
		for threat in visible_threats():
			if float(threat.distance) < 1.6 and not camp_near():
				health -= 2.0
		health = clampf(health, 0.0, 100.0)
		if health <= 0.0:
			dead = true
			last_event = "旅程结束了。记住这次经历，再开启一个清晨。"
			_record_step(maxf(0.0, health_before_tick - health))
			break
		_record_step(maxf(0.0, health_before_tick - health))

func _record_step(damage: float = 0.0) -> void:
	_reveal()
	last_step_trace.append({"tick": tick, "health": health, "hunger": hunger, "thirst": thirst, "energy": energy, "damage": damage, "player": {"x": player.x, "y": player.y}, "inventory": inventory.duplicate(), "camp": camp.duplicate(), "dead": dead, "visited_count": discovered.size()})

func _reveal() -> void:
	for y in range(player.y - VIEW_RADIUS, player.y + VIEW_RADIUS + 1):
		for x in range(player.x - VIEW_RADIUS, player.x + VIEW_RADIUS + 1):
			var p := Vector2i(x, y)
			if x >= 0 and y >= 0 and x < SIZE and y < SIZE and Vector2(p).distance_to(Vector2(player)) <= VIEW_RADIUS:
				discovered[tile_id(p)] = true

func interaction_tile(target: Vector2i) -> Vector2i:
	var path: Array[Vector2i] = path_to(target)
	return path.back() if not path.is_empty() else Vector2i(-1, -1)

func path_to(target: Vector2i) -> Array[Vector2i]:
	# A shared BFS tree amortizes all visible-target queries in one decision step.
	var empty: Array[Vector2i] = []
	if not discovered.has(tile_id(target)) and not (bool(camp.exists) and target == Vector2i(int(camp.x), int(camp.y))):
		return empty
	var key: String = "%s:%d:%d" % [player, discovered.size(), collision_revision]
	if key != _route_key:
		_route_key = key
		_route_parents = {player: player}
		_route_depths = {player: 0}
		var queue: Array[Vector2i] = [player]
		var cursor: int = 0
		while cursor < queue.size():
			var point: Vector2i = queue[cursor]
			cursor += 1
			for direction in DIRECTIONS:
				var next: Vector2i = point + direction
				if not _route_parents.has(next) and discovered.has(tile_id(next)) and is_walkable(next):
					_route_parents[next] = point
					_route_depths[next] = int(_route_depths[point]) + 1
					queue.append(next)
	var reached: Vector2i = target
	if not is_walkable(target):
		reached = Vector2i(-1, -1)
		var best_depth: int = SIZE * SIZE
		for direction in DIRECTIONS:
			var neighbor: Vector2i = target + direction
			if _route_depths.has(neighbor) and int(_route_depths[neighbor]) < best_depth:
				best_depth = int(_route_depths[neighbor])
				reached = neighbor
	if not _route_parents.has(reached):
		return empty
	var route: Array[Vector2i] = [reached]
	while reached != player:
		reached = _route_parents[reached]
		route.push_front(reached)
	return route

func step_toward(target: Vector2i) -> Dictionary:
	var path: Array[Vector2i] = path_to(target)
	if path.size() < 2:
		return {"verb": "wait"}
	return {"verb": "move", "target_id": tile_id(path[1])}

func baseline_action() -> Dictionary:
	# Explicit, inspectable scripted comparator. This is NOT a language model.
	var actions: Array[Dictionary] = allowed_actions()
	if dead:
		return {}
	if thirst < 55.0:
		if actions.has({"verb": "drink"}):
			return {"verb": "drink"}
		var nearest_water: Vector2i = _nearest_shore()
		if nearest_water.x >= 0:
			return step_toward(nearest_water)
	if hunger < 65.0 and int(inventory.berries) > 0:
		return {"verb": "eat"}
	if energy < 38.0:
		return {"verb": "rest"}
	if int(inventory.berries) < 3:
		var food: Dictionary = _seek_kind(["berry", "mushroom"])
		if not food.is_empty():
			return food
	if not bool(camp.exists):
		if actions.has({"verb": "build_camp"}):
			return {"verb": "build_camp"}
		if int(inventory.wood) < 4:
			var wood_action: Dictionary = _seek_kind(["tree", "pine"])
			if not wood_action.is_empty():
				return wood_action
		if int(inventory.stone) < 2:
			var stone_action: Dictionary = _seek_kind(["rock"])
			if not stone_action.is_empty():
				return stone_action
	if bool(camp.exists) and (is_night() or health < 80.0):
		if camp_near():
			return {"verb": "rest"}
		return step_toward(Vector2i(int(camp.x), int(camp.y)))
	# Daylight survey around the known home area, not omniscient resource selection.
	var waypoint: Vector2i = Vector2i(48, 48) + [Vector2i(3, -4), Vector2i(6, 1), Vector2i(2, 5), Vector2i(0, 0)][(tick / 14) % 4]
	var travel: Dictionary = step_toward(waypoint)
	if travel.verb != "wait":
		return travel
	return {"verb": "rest"}

func _nearest_shore() -> Vector2i:
	for landmark in navigation_landmarks():
		if str(landmark.kind) == "water":
			return Vector2i(int(landmark.x), int(landmark.y))
	return Vector2i(-1, -1)

func _seek_kind(kinds: Array) -> Dictionary:
	for r in visible_resources():
		if bool(r.available) and kinds.has(str(r.kind)):
			if can_interact(Vector2i(int(r.x), int(r.y))):
				return {"verb": "gather", "target_id": str(r.id)}
			var action: Dictionary = step_toward(Vector2i(int(r.x), int(r.y)))
			if str(action.verb) != "wait":
				return action
	return {}

