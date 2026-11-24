class_name MeadowChallengeWorld
extends "res://scripts/world_state.gd"
## Candidate environments are separate from the frozen M6 world.

const ENVIRONMENT_VERSION: String = "m7_terrain_v1"
var challenge_enabled: bool = false
var extra_energy_spent: float = 0.0
var movement_energy_cost: float = 0.0

func reset(new_seed: int = 2304, difficulty: String = "") -> void:
	var selected: String = difficulty if not difficulty.is_empty() else profile
	assert(selected in ["m7_control", "m7_terrain"], "Unknown M7 challenge profile")
	super.reset(new_seed, "open1")
	profile = selected
	challenge_enabled = profile == "m7_terrain"
	extra_energy_spent = 0.0
	movement_energy_cost = 0.0

func move_energy_at(point: Vector2i) -> float:
	if challenge_enabled:
		match terrain_at(point):
			FOREST: return 0.70
			SAND: return 1.00
	return 0.35

func adjacent_move_energy() -> Array:
	var costs: Array = []
	for direction in DIRECTIONS:
		var point: Vector2i = player + direction
		costs.append(move_energy_at(point) if discovered.has(tile_id(point)) and is_walkable(point) else null)
	return costs

func observation() -> Dictionary:
	var result: Dictionary = super.observation()
	result["environment_version"] = ENVIRONMENT_VERSION
	result["challenge"] = {"id": "terrain_fatigue_v1", "enabled": challenge_enabled}
	result["adjacent_move_energy"] = adjacent_move_energy()
	return result

func _advance(steps: int) -> void:
	movement_energy_cost = 0.0
	if last_verb in ["move", "navigate"]:
		# Parent already charged .35 after collision-safe movement. Charge only
		# the difference, before its tick damage and after its before snapshot.
		movement_energy_cost = move_energy_at(player)
		var extra: float = movement_energy_cost - 0.35
		energy = maxf(0.0, energy - extra)
		extra_energy_spent += extra
	super._advance(steps)

func _record_step(damage: float = 0.0) -> void:
	super._record_step(damage)
	last_step_trace.back()["extra_energy_spent"] = extra_energy_spent
	last_step_trace.back()["movement_energy_cost"] = movement_energy_cost
