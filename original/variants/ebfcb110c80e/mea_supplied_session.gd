extends "res://scripts/m9_environment_session.gd"
## Offline crossover replay. The M9 map and action rules come from the parent.
const SCENE_NAME := "stress_thirst"

func handle(request: Dictionary) -> Dictionary:
	if request.get("op", "") == "landmark":
		return {"ok": true, "id": request.id, "landmark": world.archive_landmark}
	var response := super.handle(request)
	if not response.get("ok", false):
		return response
	if request.get("op", "") == "reset":
		world.health = 25.0
		world.thirst = 1.6
		world.hunger = 95.0
		response = super.handle({"op": "observe", "id": request.id})
	response["starting_scene"] = SCENE_NAME
	return response
