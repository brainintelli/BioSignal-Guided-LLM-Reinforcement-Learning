class_name MeadowSurvivalWorld
extends "res://scripts/world_state.gd"
## Current three-vital world. Legacy rules remain in separately versioned worlds.
const SURVIVAL_WORLD_VERSION: int = 6
const ENVIRONMENT_VERSION: String = "m8_survival_v2"
var balance: Dictionary = {}
var balance_sha256: String = ""
var rules_valid: bool = false

func _init() -> void:
	var file := FileAccess.open("res://configs/m8_survival_v2.json", FileAccess.READ)
	if file == null: return
	var bytes: PackedByteArray = file.get_buffer(file.get_length())
	var digest := HashingContext.new()
	digest.start(HashingContext.HASH_SHA256)
	digest.update(bytes)
	balance_sha256 = digest.finish().hex_encode()
	var parsed: Variant = JSON.parse_string(bytes.get_string_from_utf8())
	if parsed is Dictionary: balance = parsed

func validation_error() -> String:
	var expected: Dictionary = {"environment_version": "m8_survival_v2", "world_version": 6, "profile": "open4", "day_ticks": 180, "hunger_drain": 0.2, "thirst_drain": 0.08, "drink_restore": 50, "food_restore": 18, "food_gather_count": 3, "resource_regrowth_ticks": 90, "rest_ticks": 3, "camp_rest_health": 2, "heal_min_hunger": 20, "heal_min_thirst": 20, "threat_damage": 1.2, "hunger_damage": 1.2, "thirst_damage": 1.8, "initial_health": 100, "initial_hunger": 82, "initial_thirst": 78, "initial_berries": 2, "camp_wood": 4, "camp_stone": 2, "camp_radius": 2.5}
	if balance.size() != expected.size(): return "survival_parameter_set_mismatch"
	for key in expected:
		if expected[key] is String:
			if not balance.get(key) is String or balance[key] != expected[key]: return "survival_identity_mismatch_" + key
		else:
			if not (balance.get(key) is int or balance.get(key) is float): return "invalid_parameter_" + key
			if not is_finite(float(balance[key])) or float(balance[key]) != float(expected[key]): return "changed_parameter_" + key
	return ""

func reset(new_seed: int = 2304, difficulty: String = "open4") -> void:
	rules_valid = difficulty == "open4" and validation_error().is_empty()
	if not rules_valid:
		dead = true
		return
	# Reuse only the deterministic map initialization; no legacy action executes.
	super.reset(new_seed, "open1")
	profile = "open4"

func allowed_actions() -> Array[Dictionary]:
	if not rules_valid: return []
	return super.allowed_actions()

func observation() -> Dictionary:
	# Known-map route metadata is retained for external reward accounting only.
	# The actor encoder explicitly excludes it from its state and prompt.
	var landmarks: Array[Dictionary] = navigation_landmarks()
	for landmark in landmarks:
		var route: Array[Vector2i] = path_to(Vector2i(int(landmark.x), int(landmark.y)))
		landmark["path_steps"] = route.size() - 1 if not route.is_empty() else -1
	return {"schema_version": 1, "world_version": SURVIVAL_WORLD_VERSION, "profile": profile,
		"environment_version": ENVIRONMENT_VERSION, "balance_sha256": balance_sha256,
		"tick": tick, "day": day(), "phase": phase(),
		"player": {"health": snappedf(health, 0.1), "hunger": snappedf(hunger, 0.1), "thirst": snappedf(thirst, 0.1), "x": player.x, "y": player.y},
		"inventory": inventory.duplicate(), "nearby": landmarks, "threats": visible_threats(),
		"local_map": local_observation_map(), "local_visits": local_visit_counts(),
		"camp": camp.duplicate(), "allowed_actions": allowed_actions(), "visited_count": discovered.size(),
		"recent_actions": recent_actions.duplicate(true), "navigation_assistance": "selected_target_one_collision_safe_step"}

func apply_action(action: Dictionary) -> bool:
	last_step_trace.clear()
	if not rules_valid or dead or not allowed_actions().has(action):
		return false
	var before: Dictionary = {"health": health, "hunger": hunger, "thirst": thirst}
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
			gathered += 1
			cost = 2
		"eat":
			inventory.berries = int(inventory.berries) - 1
			hunger = minf(100.0, hunger + 18.0)
			last_event = "吃下一份野果，饱食 +18。"
		"drink":
			thirst = minf(100.0, thirst + float(balance.drink_restore))
			last_event = "在清澈的岸边饮水，水分 +50。"
		"build_camp":
			inventory.wood = int(inventory.wood) - 4
			inventory.stone = int(inventory.stone) - 2
			var site: Vector2i = camp_placement()
			camp = {"exists": true, "x": site.x, "y": site.y}
			collision_revision += 1
			last_event = "营地建好了！在附近休息更安全，也能恢复生命。"
			cost = 4
		"rest":
			if camp_near() and hunger >= float(balance.heal_min_hunger) and thirst >= float(balance.heal_min_thirst):
				health = minf(100.0, health + float(balance.camp_rest_health))
			last_event = "在营地休息；食水充足时恢复生命。" if camp_near() else "坐下来休息三刻。"
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
			"thirst": snappedf(thirst - float(before.thirst), 0.1)}, "tick": tick})
	if recent_actions.size() > 4:
		recent_actions.pop_front()
	return true

func _advance(steps: int) -> void:
	for _i in range(steps):
		var health_before_tick: float = health
		tick += 1
		hunger = maxf(0.0, hunger - float(balance.hunger_drain))
		thirst = maxf(0.0, thirst - float(balance.thirst_drain))
		if hunger <= 0.0: health -= float(balance.hunger_damage)
		if thirst <= 0.0: health -= float(balance.thirst_damage)
		# Darkness itself never damages the player. Nearby threats still do.
		for threat in visible_threats():
			if float(threat.distance) < 1.6 and not camp_near(): health -= float(balance.threat_damage)
		health = clampf(health, 0.0, 100.0)
		if health <= 0.0:
			dead = true
			last_event = "旅程结束了。记住这次经历，再开启一个清晨。"
			_record_step(maxf(0.0, health_before_tick - health))
			break
		_record_step(maxf(0.0, health_before_tick - health))

func _record_step(damage: float = 0.0) -> void:
	_reveal()
	last_step_trace.append({"tick": tick, "health": health, "hunger": hunger, "thirst": thirst, "damage": damage, "player": {"x": player.x, "y": player.y}, "inventory": inventory.duplicate(), "camp": camp.duplicate(), "dead": dead, "visited_count": discovered.size()})

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
	if bool(camp.exists) and health < 80.0:
		if camp_near():
			return {"verb": "rest"}
		return step_toward(Vector2i(int(camp.x), int(camp.y)))
	# Survey the known home area without using nighttime or a removed vital.
	var waypoint: Vector2i = Vector2i(48, 48) + [Vector2i(3, -4), Vector2i(6, 1), Vector2i(2, 5), Vector2i(0, 0)][(tick / 14) % 4]
	var travel: Dictionary = step_toward(waypoint)
	if travel.verb != "wait":
		return travel
	return {"verb": "wait"}

