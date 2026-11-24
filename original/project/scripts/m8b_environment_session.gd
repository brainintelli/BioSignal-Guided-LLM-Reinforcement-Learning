extends "res://scripts/rl_environment_session.gd"
const BalancedWorld = preload("res://scripts/balanced_world.gd")
func _init() -> void:
 world = BalancedWorld.new()
func handle(request: Dictionary) -> Dictionary:
 var response: Dictionary
 if str(request.get("op", "")) == "reset":
  var request_id: Variant = request.get("id", null)
  if str(request.get("profile", "")) != "open2":
   return _error(request_id, "unsupported_balanced_profile")
  var seed_value: Variant = request.get("seed", 2304)
  if not (seed_value is int or seed_value is float) or not is_finite(float(seed_value)) or float(seed_value) != float(int(seed_value)):
   return _error(request_id, "seed_must_be_integer")
  world.reset(int(seed_value), "open2")
  if not world.rules_valid:
   initialized = false
   return _error(request_id, "invalid_balance_configuration:" + world.validation_error())
  initialized = true
  response = _reply(request_id)
 else:
  response = super.handle(request)
 response["world_version"] = 4
 response["environment_version"] = BalancedWorld.ENVIRONMENT_VERSION
 response["profile"] = "open2"
 response["balance_sha256"] = world.balance_sha256
 if response.has("transition"):
  response.transition["world_version"] = 4
  response.transition["environment_version"] = BalancedWorld.ENVIRONMENT_VERSION
  response.transition["profile"] = "open2"
  response.transition["balance_sha256"] = world.balance_sha256
 return response
func snapshot() -> Dictionary:
 var result: Dictionary = super.snapshot()
 result["world_version"] = 4
 result["environment_version"] = BalancedWorld.ENVIRONMENT_VERSION
 result["profile"] = "open2"
 result["balance_sha256"] = world.balance_sha256
 return result
