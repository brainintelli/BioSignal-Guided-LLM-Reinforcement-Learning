extends "res://scripts/m9_environment_session.gd"
## Opt-in M9 curriculum & stress-suite session.
## Preserves exact M9 survival rules, balance_sha256, and map generation; only allows
## explicitly registered initial vital/clock/position overrides on `reset`.

var current_scenario: String = "standard"

func _place_near_slime() -> bool:
	if not world.is_night():
		return false
	var target: Vector2i = world._layout_point(Vector2i(51, 47))
	var best := Vector2i(-1, -1)
	var best_dist: float = 9999.0
	for dy in range(-1, 2):
		for dx in range(-1, 2):
			if dx == 0 and dy == 0:
				continue
			var cand := target + Vector2i(dx, dy)
			var dist: float = Vector2(cand).distance_to(Vector2(target))
			if dist < 1.5 and world.is_walkable(cand):
				var pdist: float = Vector2(cand).distance_to(Vector2(world.player))
				if pdist < best_dist:
					best_dist = pdist
					best = cand
	if best.x < 0:
		return false
	world.player = best
	if world.streaming:
		world.activate_region()
	world.discovered.clear()
	world.entered_counts.clear()
	world.entered_counts[world.tile_id(world.player)] = 1
	world._reveal()
	return true

func handle(request: Dictionary) -> Dictionary:
	var response: Dictionary = super.handle(request)
	if not response.get("ok", false):
		return response
	if str(request.get("op", "")) == "reset":
		current_scenario = str(request.get("scenario_name", "standard"))
		var changed: bool = false
		if request.has("initial_health") and request.get("initial_health") != null:
			world.health = clampf(float(request.get("initial_health")), 1.0, 100.0)
			changed = true
		if request.has("initial_thirst") and request.get("initial_thirst") != null:
			world.thirst = clampf(float(request.get("initial_thirst")), 0.1, 100.0)
			changed = true
		if request.has("initial_hunger") and request.get("initial_hunger") != null:
			world.hunger = clampf(float(request.get("initial_hunger")), 0.1, 100.0)
			changed = true
		if request.has("initial_berries") and request.get("initial_berries") != null:
			world.inventory["berries"] = maxi(0, int(request.get("initial_berries")))
			changed = true
		if request.has("initial_tick") and request.get("initial_tick") != null:
			world.tick = maxi(0, int(request.get("initial_tick")))
			if world.streaming:
				world._sync_deadlines()
			changed = true
		if bool(request.get("near_threat", false)):
			if _place_near_slime():
				changed = true
		if changed:
			response = super.handle({"op": "observe", "id": request.get("id", null)})
	response["scenario_name"] = current_scenario
	return response
