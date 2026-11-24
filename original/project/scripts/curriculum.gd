extends RefCounted
## Controlled teaching scenarios, not model weight training. All actions stay legal.
const VERSION: int = 1
var _state: Dictionary = {}
var _before: Dictionary = {}

static func metadata() -> Array[Dictionary]:
	return [
		{"id": "eat", "title": "01 吃一口野果", "goal": "从背包吃一份食物，恢复饱食", "budget_world_steps": 8},
		{"id": "drink", "title": "02 岸边饮水", "goal": "在岸边饮水，恢复水分", "budget_world_steps": 8},
		{"id": "fetch_water", "title": "03 走向水源", "goal": "走到岸边并饮水", "budget_world_steps": 24},
		{"id": "forage", "title": "04 自己找食物", "goal": "先采集食物，再吃一份", "budget_world_steps": 24},
		{"id": "camp_ready", "title": "05 搭起营地", "goal": "用已有的 4 木材、2 石块建营", "budget_world_steps": 12},
		{"id": "camp_chain", "title": "06 从零扎营", "goal": "采集 4 木材与 2 石块，再建营", "budget_world_steps": 64}
	]

func configure(world: MeadowWorld, id: String, seed: int) -> bool:
	_state = {}
	for item in metadata():
		if str(item.id) == id:
			_state = item.duplicate(true)
	if _state.is_empty():
		return false
	world.reset(seed, "d0")
	# Only this 15x13 clearing is controlled; the rest of the 96x96 world survives.
	for y in range(42, 55):
		for x in range(41, 56):
			var p := Vector2i(x, y)
			world.terrain[y * world.SIZE + x] = world.GRASS
			world._remove_resource_at(p)
	for y in range(43, 54):
		for x in range(42, 44):
			world.terrain[y * world.SIZE + x] = world.WATER
	world.player = Vector2i(48, 48)
	world.inventory = {"wood": 0, "stone": 0, "berries": 0}
	world._add_resource(Vector2i(50, 48), "berry")
	world._add_resource(Vector2i(49, 46), "tree")
	world._add_resource(Vector2i(51, 47), "pine")
	world._add_resource(Vector2i(50, 50), "rock")
	if id in ["eat", "forage"]:
		world.hunger = 25.0
	if id == "eat":
		world.inventory.berries = 2
	if id in ["drink", "fetch_water"]:
		world.thirst = 25.0
	if id == "drink":
		world.player = Vector2i(44, 48)
	if id == "camp_ready":
		world.inventory.wood = 4
		world.inventory.stone = 2
	world.discovered.clear()
	world.rebuild_spatial_index()
	world._reveal()
	world.last_event = "受控初始条件 · D0 教学场景 · 非权重训练"
	_state.merge({"curriculum_version": VERSION, "elapsed": 0, "status": "active", "source": "none", "sources": [], "source_counts": {}, "initial": {"fixture": "clearing_v1_15x13", "profile": "d0", "seed": seed, "player": {"x": world.player.x, "y": world.player.y}, "inventory": world.inventory.duplicate(), "hunger": world.hunger, "thirst": world.thirst}, "progress": {"food_gathered": 0, "wood_gathered": 0, "stone_gathered": 0, "moves": 0}, "controlled_initial_conditions": true})
	_before = _snapshot(world)
	return true

func state() -> Dictionary:
	return _state.duplicate(true)

func observation(world: MeadowWorld) -> Dictionary:
	var result: Dictionary = world.observation()
	if not _state.is_empty():
		result.task = {}
		for key in ["id", "title", "goal", "budget_world_steps", "elapsed", "status", "progress", "curriculum_version", "controlled_initial_conditions"]:
			result.task[key] = _state[key].duplicate(true) if _state[key] is Dictionary else _state[key]
	return result

func _snapshot(world: MeadowWorld) -> Dictionary:
	return {"tick": world.tick, "inventory": world.inventory.duplicate(), "hunger": world.hunger, "thirst": world.thirst, "camp": bool(world.camp.exists), "player": world.player}

func after_action(world: MeadowWorld, action: Dictionary, source: String) -> void:
	if _state.is_empty() or str(_state.status) != "active" or world.tick <= int(_before.tick):
		return
	# Cross-check the actual simulation record, not merely a claimed action.
	var actual: bool = not world.recent_actions.is_empty() and world.recent_actions.back().action == action and int(world.recent_actions.back().tick) == world.tick
	_state.elapsed = world.tick
	var progress: Dictionary = _state.progress
	var verb: String = str(action.get("verb", "")) if actual else ""
	var prior: Dictionary = _before.inventory
	if actual:
		_state.source_counts[source] = int(_state.source_counts.get(source, 0)) + 1
		if not _state.sources.has(source):
			_state.sources.append(source)
		_state.source = source if _state.sources.size() == 1 else "mixed"
	if verb in ["move", "navigate"] and world.player != _before.player:
		progress.moves += 1
	if verb == "gather":
		progress.food_gathered += maxi(0, int(world.inventory.berries) - int(prior.berries))
		progress.wood_gathered += maxi(0, int(world.inventory.wood) - int(prior.wood))
		progress.stone_gathered += maxi(0, int(world.inventory.stone) - int(prior.stone))
	var ate: bool = verb == "eat" and int(world.inventory.berries) == int(prior.berries) - 1 and world.hunger > float(_before.hunger)
	var drank: bool = verb == "drink" and world.thirst > float(_before.thirst)
	var built: bool = verb == "build_camp" and not bool(_before.camp) and bool(world.camp.exists) and int(prior.wood) - int(world.inventory.wood) == 4 and int(prior.stone) - int(world.inventory.stone) == 2
	var succeeded: bool = false
	match str(_state.id):
		"eat": succeeded = ate
		"drink": succeeded = drank
		"fetch_water": succeeded = drank and int(progress.moves) >= 3
		"forage": succeeded = ate and int(progress.food_gathered) > 0
		"camp_ready": succeeded = built
		"camp_chain": succeeded = built and int(progress.wood_gathered) >= 4 and int(progress.stone_gathered) >= 2
	if world.dead or int(_state.elapsed) > int(_state.budget_world_steps):
		_state.status = "failed"
	elif succeeded:
		_state.status = "success"
	elif int(_state.elapsed) >= int(_state.budget_world_steps):
		_state.status = "failed"
	_before = _snapshot(world)

func baseline_action(world: MeadowWorld) -> Dictionary:
	# Explicit comparator consuming exactly the same visible observation and whitelist.
	var obs: Dictionary = observation(world)
	var actions: Array = obs.allowed_actions
	var desired: String = "eat"
	var kinds: Array = ["berry", "mushroom"]
	if str(_state.get("id", "")) in ["drink", "fetch_water"]:
		desired = "drink"
		kinds = ["water"]
	elif str(_state.get("id", "")) in ["camp_ready", "camp_chain"]:
		desired = "build_camp"
		kinds = ["tree", "pine"] if int(obs.inventory.wood) < 4 else ["rock"]
	if actions.has({"verb": desired}):
		return {"verb": desired}
	for nearby in obs.nearby:
		if not bool(nearby.available) or not kinds.has(str(nearby.kind)):
			continue
		for verb in ["gather", "navigate"]:
			var action: Dictionary = {"verb": verb, "target_id": str(nearby.id)}
			if actions.has(action):
				return action
	return {"verb": "wait"}

