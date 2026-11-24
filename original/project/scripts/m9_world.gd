extends "res://scripts/survival_world.gd"
## Research world: native survival dynamics, optional on-demand 64-tile regions.
## Coordinate limit is explicit; visited chunk records remain in RAM for this bounded campaign.
const CHUNK_SIDE: int = 64
const COORD_LIMIT: int = 1000000
var map_mode: String = "finite"
var navigation_steps: int = 1
var streaming: bool = false
var chunks: Dictionary = {}
var active_region: Vector2i = Vector2i(2147483647,2147483647)
var origin_resources: Dictionary = {}

func configure(mode: String, steps: int) -> void:
	assert(mode in ["finite","infinite_freeze","infinite_global"])
	assert(steps in [1,4])
	map_mode = mode
	navigation_steps = steps

func reset(new_seed: int = 2304, difficulty: String = "m9") -> void:
	streaming = false
	chunks.clear(); origin_resources.clear()
	active_region = Vector2i(2147483647,2147483647)
	super.reset(new_seed, "open4")
	if difficulty != "m9":
		rules_valid = false
		dead = true
		return
	profile = "m9"
	for r in resources:
		origin_resources[Vector2i(int(r.x),int(r.y))] = r.duplicate(true)
	streaming = map_mode != "finite"
	if streaming: activate_region()
	_reveal()

func chunk_of(p: Vector2i) -> Vector2i:
	return Vector2i(floori(float(p.x)/CHUNK_SIDE),floori(float(p.y)/CHUNK_SIDE))

func _hash(x: int, y: int, tag: String) -> int:
	return ("%d:%d:%d:%s" % [world_seed,x,y,tag]).sha256_text().substr(0,8).hex_to_int()

func _generated_terrain(p: Vector2i) -> int:
	# Integer-cell seed, then only small local coordinates enter the pond equation.
	var cell := Vector2i(floori(float(p.x)/32),floori(float(p.y)/32))
	var h: int = _hash(cell.x,cell.y,"pond")
	var cx: int = cell.x*32+8+int(h%16)
	var cy: int = cell.y*32+8+int((h/16)%16)
	var dx: float = float(p.x-cx)/4.0
	var dy: float = float(p.y-cy)/4.0
	var lake: float = dx*dx+dy*dy
	if lake < 1.0: return WATER
	if lake < 1.65: return SAND
	var value: int = _hash(p.x,p.y,"ground")%100
	return FOREST if value < 20 else (FLOWERS if value < 35 else GRASS)

func ensure_chunk(coord: Vector2i) -> void:
	if chunks.has(coord): return
	var tiles := PackedInt32Array()
	tiles.resize(CHUNK_SIDE*CHUNK_SIDE)
	var objects: Array[Dictionary] = []
	for y in range(CHUNK_SIDE):
		for x in range(CHUNK_SIDE):
			var p: Vector2i = coord*CHUNK_SIDE+Vector2i(x,y)
			var original: bool = p.x>=0 and p.y>=0 and p.x<96 and p.y<96
			var kind: int = terrain[p.y*96+p.x] if original else _generated_terrain(p)
			tiles[y*CHUNK_SIDE+x] = kind
			var obj: Dictionary = {}
			if original:
				if origin_resources.has(p): obj = origin_resources[p].duplicate(true)
			elif kind != WATER and _hash(p.x,p.y,"resource")%1000 < 85:
				var kinds: Array[String] = ["tree","pine","berry","rock","mushroom"]
				obj = {"id":"r_%d_%d"%[p.x,p.y],"kind":kinds[_hash(p.x,p.y,"kind")%5],"x":p.x,"y":p.y,"ready_at":0}
			if not obj.is_empty():
				obj["ready_local"] = 0
				objects.append(obj)
	chunks[coord] = {"local_tick":0,"tiles":tiles,"resources":objects}

func activate_region() -> void:
	if not streaming: return
	var center: Vector2i = chunk_of(player)
	if center != active_region:
		active_region = center
		resources.clear()
		for dy in range(-1,2):
			for dx in range(-1,2):
				var coord: Vector2i = center+Vector2i(dx,dy)
				ensure_chunk(coord)
				for r in chunks[coord].resources: resources.append(r)
		rebuild_spatial_index()
		_route_key = ""
	_sync_deadlines()

func _sync_deadlines() -> void:
	if not streaming or map_mode != "infinite_freeze": return
	for dy in range(-1,2):
		for dx in range(-1,2):
			var chunk: Dictionary = chunks[active_region+Vector2i(dx,dy)]
			for r in chunk.resources:
				r.ready_at = tick+maxi(0,int(r.ready_local)-int(chunk.local_tick))

func terrain_at(p: Vector2i) -> int:
	if not streaming: return super.terrain_at(p)
	if absi(p.x)>COORD_LIMIT or absi(p.y)>COORD_LIMIT: return WATER
	var coord: Vector2i = chunk_of(p)
	ensure_chunk(coord)
	var local: Vector2i = p-coord*CHUNK_SIDE
	return int(chunks[coord].tiles[local.y*CHUNK_SIDE+local.x])

func is_walkable(p: Vector2i) -> bool:
	if not streaming: return super.is_walkable(p)
	return absi(p.x)<COORD_LIMIT and absi(p.y)<COORD_LIMIT and footprint_at(p).is_empty()

func footprint_at(p: Vector2i) -> Dictionary:
	if not streaming: return super.footprint_at(p)
	if absi(p.x)>=COORD_LIMIT or absi(p.y)>=COORD_LIMIT:
		return _footprint_geometry("coordinate_limit", p)
	if terrain_at(p)==WATER: return _footprint_geometry("water", p)
	if bool(camp.exists) and p==Vector2i(int(camp.x),int(camp.y)): return _footprint_geometry("camp", p)
	var r: Dictionary = resource_at(p)
	return {} if r.is_empty() else _footprint_geometry(str(r.kind), p)

func collision_footprints() -> Array[Dictionary]:
	if not streaming: return super.collision_footprints()
	var result: Array[Dictionary] = []
	var region_start: Vector2i = (active_region - Vector2i.ONE) * CHUNK_SIDE
	var region_end: Vector2i = region_start + Vector2i.ONE * (3 * CHUNK_SIDE)
	for y in range(region_start.y, region_end.y):
		var x: int = region_start.x
		while x < region_end.x:
			if terrain_at(Vector2i(x, y)) != WATER:
				x += 1
				continue
			var start: int = x
			while x < region_end.x and terrain_at(Vector2i(x, y)) == WATER:
				x += 1
			result.append({"shape": "rect", "center_px": Vector2(float(start + x - 1) * 0.5, y) * TILE_SIZE,
				"size": Vector2(float(x - start) * TILE_SIZE, TILE_SIZE), "kind": "water"})
	for resource in resources:
		var shape: Dictionary = footprint_at(Vector2i(int(resource.x), int(resource.y)))
		if not shape.is_empty() and shape.kind != "water":
			result.append(shape)
	if bool(camp.exists):
		result.append(_footprint_geometry("camp", Vector2i(int(camp.x), int(camp.y))))
	return result

func allowed_actions() -> Array[Dictionary]:
	var result: Array[Dictionary] = super.allowed_actions()
	if not streaming or dead or not rules_valid: return result
	result.erase({"verb":"build_camp"})
	if not bool(camp.exists) and int(inventory.wood)>=4 and int(inventory.stone)>=2:
		var exits: int = 0
		var free_site: bool = false
		for direction in DIRECTIONS:
			var p: Vector2i = player+direction
			if is_walkable(p):
				exits+=1
				if resource_at(p).is_empty(): free_site=true
		if exits>=2 and free_site: result.append({"verb":"build_camp"})
	return result

func _reveal() -> void:
	if not streaming:
		super._reveal()
		return
	activate_region()
	for y in range(player.y-VIEW_RADIUS,player.y+VIEW_RADIUS+1):
		for x in range(player.x-VIEW_RADIUS,player.x+VIEW_RADIUS+1):
			var p := Vector2i(x,y)
			if Vector2(p).distance_to(Vector2(player))<=VIEW_RADIUS:
				discovered[tile_id(p)] = true

func path_to(target: Vector2i) -> Array[Vector2i]:
	if not streaming: return super.path_to(target)
	# Known cells only, bounded local BFS; never search an ever-growing world graph.
	var empty: Array[Vector2i] = []
	if not discovered.has(tile_id(target)): return empty
	var key: String = "%s:%d:%d" % [player,discovered.size(),collision_revision]
	if key != _route_key:
		_route_key = key
		_route_parents = {player:player}; _route_depths = {player:0}
		var queue: Array[Vector2i] = [player]
		var cursor: int = 0
		while cursor<queue.size():
			var p: Vector2i = queue[cursor]; cursor+=1
			for direction in DIRECTIONS:
				var n: Vector2i = p+direction
				if maxi(absi(n.x-player.x),absi(n.y-player.y))>16: continue
				if not _route_parents.has(n) and discovered.has(tile_id(n)) and is_walkable(n):
					_route_parents[n]=p; _route_depths[n]=int(_route_depths[p])+1
					queue.append(n)
	var reached: Vector2i = target
	if not is_walkable(target):
		var best: int = 1000000
		for direction in DIRECTIONS:
			var neighbor: Vector2i = target+direction
			if _route_depths.has(neighbor) and int(_route_depths[neighbor])<best:
				best=int(_route_depths[neighbor]); reached=neighbor
	if not _route_parents.has(reached): return empty
	var route: Array[Vector2i] = [reached]
	while reached!=player:
		reached=_route_parents[reached]; route.push_front(reached)
	return route

func _advance(steps: int) -> void:
	if not streaming:
		super._advance(steps)
		return
	for i in range(steps):
		activate_region()
		for dy in range(-1,2):
			for dx in range(-1,2):
				chunks[active_region+Vector2i(dx,dy)].local_tick += 1
		super._advance(1)
		_sync_deadlines()
		if dead: break

func apply_action(action: Dictionary) -> bool:
	if not allowed_actions().has(action) or dead: return false
	var trace: Array = []
	var repeats: int = navigation_steps if action.verb=="navigate" else 1
	if streaming and action.verb=="gather" and map_mode=="infinite_freeze":
		var r: Dictionary = resource_by_id(str(action.target_id))
		var coord: Vector2i = chunk_of(Vector2i(int(r.x),int(r.y)))
		r.ready_local = int(chunks[coord].local_tick)+90
	for i in range(repeats):
		if dead or not allowed_actions().has(action): break
		if not super.apply_action(action): return false
		trace.append_array(last_step_trace.duplicate(true))
	last_step_trace.assign(trace)
	return not trace.is_empty()

func observation() -> Dictionary:
	activate_region()
	var result: Dictionary = super.observation()
	result["profile"]="m9"
	result["world_version"]=7
	result["environment_version"]="m9_regions_v1"
	result["map_mode"]=map_mode
	result["navigation_steps"]=navigation_steps
	result["world_tick"]=tick
	result["navigation_assistance"]="selected_target_up_to_4_collision_safe_steps" if navigation_steps==4 else "selected_target_one_collision_safe_step"
	return result

