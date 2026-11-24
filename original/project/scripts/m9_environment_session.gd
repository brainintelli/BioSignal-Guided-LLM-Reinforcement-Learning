extends "res://scripts/m8s3_environment_session.gd"
const RegionWorld = preload("res://scripts/m9_world.gd")
func _init() -> void:
	world = RegionWorld.new()
func handle(request: Dictionary) -> Dictionary:
	var response: Dictionary
	if str(request.get("op","")) == "reset":
		var rid: Variant = request.get("id",null)
		var mode: String = str(request.get("map_mode","finite"))
		var steps: int = int(request.get("navigation_steps",1))
		if str(request.get("profile",""))!="m9" or mode not in ["finite","infinite_freeze","infinite_global"] or steps not in [1,4]:
			return _error(rid,"unregistered_m9_configuration")
		var seed_value: Variant = request.get("seed",0)
		if not (seed_value is int or seed_value is float) or not is_finite(float(seed_value)) or float(seed_value)!=float(int(seed_value)):
			return _error(rid,"seed_must_be_integer")
		world.configure(mode,steps)
		world.reset(int(seed_value),"m9")
		initialized=world.rules_valid
		if not initialized: return _error(rid,"invalid_survival_rules")
		response=_reply(rid)
	else:
		response=super.handle(request)
	for k in identity(): response[k]=identity()[k]
	if response.has("transition"):
		for k in identity(): response.transition[k]=identity()[k]
	return response
func identity() -> Dictionary:
	return {"world_version":7,"profile":"m9","environment_version":"m9_regions_v1",
		"balance_sha256":world.balance_sha256,"map_mode":world.map_mode,"navigation_steps":world.navigation_steps}
func snapshot() -> Dictionary:
	var result: Dictionary = super.snapshot()
	for k in identity(): result[k]=identity()[k]
	return result

