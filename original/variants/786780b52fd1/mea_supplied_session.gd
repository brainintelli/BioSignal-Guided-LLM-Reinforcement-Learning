extends "res://scripts/m9_environment_session.gd"
## Offline crossover replay. The M9 map and action rules come from the parent.
const SCENE_NAME := "thirst_challenge_25_1p6"

func handle(request: Dictionary) -> Dictionary:
	var response := super.handle(request)
	if not response.get("ok", false):
		return response
	if request.get("op", "") == "reset":
		world.hunger = 95.0
		world.thirst = 1.6
		world.health = 25.0
		response = super.handle({"op": "observe", "id": request.id})
	response["starting_scene"] = SCENE_NAME
	return response
