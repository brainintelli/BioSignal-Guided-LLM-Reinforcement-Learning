extends RefCounted
## Reward-free transport over the authoritative game, independent of rendering.
## No curriculum, suggested action, task goal, or policy is supplied here.

const World = preload("res://scripts/world_state.gd")
const PROTOCOL_VERSION: String = "meadow_rl_v1"
var world: World = World.new()
var initialized: bool = false

func handle(request: Dictionary) -> Dictionary:
	var request_id: Variant = request.get("id", null)
	var operation: String = str(request.get("op", ""))
	if operation == "close":
		return {"id": request_id, "ok": true, "closed": true, "protocol": PROTOCOL_VERSION}
	if operation == "reset":
		var difficulty: String = str(request.get("profile", "d1"))
		if difficulty not in ["d0", "d1", "open1"]:
			return _error(request_id, "unsupported_profile")
		var seed_value: Variant = request.get("seed", 2304)
		if not (seed_value is int or seed_value is float) or not is_finite(float(seed_value)) or float(seed_value) != float(int(seed_value)):
			return _error(request_id, "seed_must_be_integer")
		world.reset(int(seed_value), difficulty)
		initialized = true
		return _reply(request_id)
	if not initialized:
		return _error(request_id, "reset_required")
	if operation == "observe":
		return _reply(request_id)
	if operation != "step":
		return _error(request_id, "unknown_operation")
	if not request.get("action", null) is Dictionary:
		return _error(request_id, "action_must_be_object")
	var action: Dictionary = request.action
	if world.dead:
		return _error(request_id, "episode_terminated")
	if not world.allowed_actions().has(action):
		return _error(request_id, "illegal_action")
	var before: Dictionary = snapshot()
	if not world.apply_action(action):
		return _error(request_id, "action_rejected")
	var after: Dictionary = snapshot()
	var response: Dictionary = _reply(request_id)
	response["transition"] = {"before": before, "after": after,
		"elapsed_ticks": world.tick - int(before.tick),
		"new_discovered": world.discovered.size() - int(before.visited_count),
		"trace": world.last_step_trace.duplicate(true), "action": action.duplicate(true)}
	return response

func snapshot() -> Dictionary:
	# Exact execution facts for externally defined rewards, not extra policy input.
	return {"tick": world.tick, "health": world.health, "hunger": world.hunger,
		"thirst": world.thirst, "energy": world.energy,
		"player": {"x": world.player.x, "y": world.player.y},
		"inventory": world.inventory.duplicate(), "camp": world.camp.duplicate(),
		"visited_count": world.discovered.size(), "dead": world.dead}

func _reply(request_id: Variant) -> Dictionary:
	return {"id": request_id, "ok": true, "protocol": PROTOCOL_VERSION,
		"observation": world.observation(), "terminated": world.dead}

func _error(request_id: Variant, reason: String) -> Dictionary:
	return {"id": request_id, "ok": false, "protocol": PROTOCOL_VERSION, "error": reason}
