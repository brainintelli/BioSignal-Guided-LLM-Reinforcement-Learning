class_name MeadowExplorationWorld
extends "res://scripts/world_state.gd"
## Development exploration rules; older registered worlds remain byte-identical.
const EXPLORATION_WORLD_VERSION: int = 5
const ENVIRONMENT_VERSION: String = "m8_exploration_v1"
var action_start: Dictionary = {}
var balance: Dictionary = {}
var balance_sha256: String = ""
var rules_valid: bool = false

func _init() -> void:
 var file := FileAccess.open("res://configs/m8_exploration_v1.json", FileAccess.READ)
 if file == null:return
 var bytes: PackedByteArray = file.get_buffer(file.get_length())
 var digest := HashingContext.new()
 digest.start(HashingContext.HASH_SHA256);digest.update(bytes)
 balance_sha256 = digest.finish().hex_encode()
 var parsed: Variant = JSON.parse_string(bytes.get_string_from_utf8())
 if parsed is Dictionary:balance = parsed

func validation_error() -> String:
 if balance.get("environment_version") != ENVIRONMENT_VERSION or balance.get("world_version") != 5 or balance.get("profile") != "open3":
  return "balance_identity_mismatch"
 var inherited: Dictionary = {"day_ticks":180,"food_restore":18,"food_gather_count":3,"resource_regrowth_ticks":90,"move_energy":0.35,"gather_energy":2,"rest_energy":14,"rest_ticks":3,"initial_health":100,"initial_hunger":82,"initial_thirst":78,"initial_energy":92,"initial_berries":2,"camp_wood":4,"camp_stone":2,"camp_radius":2.5}
 for key in inherited:
  if not balance.get(key) is float and not balance.get(key) is int:return "invalid_inherited_" + key
  if not is_finite(float(balance[key])) or float(balance[key]) != float(inherited[key]):return "changed_inherited_" + key
 for key in ["hunger_drain","thirst_drain","drink_restore","camp_rest_health","heal_min_hunger","heal_min_thirst","night_damage","threat_damage","hunger_damage","thirst_damage","exhaustion_damage"]:
  if not balance.get(key) is float and not balance.get(key) is int:return "invalid_parameter_" + key
  if not is_finite(float(balance[key])) or float(balance[key]) <= 0.0:return "invalid_parameter_" + key
 var calibrated: Dictionary = {"hunger_drain":0.20,"thirst_drain":0.08,"drink_restore":50,"camp_rest_health":2,"heal_min_hunger":20,"heal_min_thirst":20,"night_damage":0.10,"threat_damage":1.2,"hunger_damage":1.2,"thirst_damage":1.8,"exhaustion_damage":0.6}
 for key in calibrated:
  if float(balance[key]) != float(calibrated[key]):return "changed_calibrated_" + key
 if float(balance.thirst_drain) >= float(balance.hunger_drain):return "water_drain_must_be_lower"
 return ""

func allowed_actions() -> Array[Dictionary]:
 if not rules_valid:return []
 return super.allowed_actions()

func reset(new_seed: int = 2304, difficulty: String = "open3") -> void:
 rules_valid = difficulty == "open3" and validation_error().is_empty()
 if not rules_valid:
  dead = true
  return
 super.reset(new_seed, "open1")
 profile = "open3"
 action_start.clear()

func observation() -> Dictionary:
 var result: Dictionary = super.observation()
 result["world_version"] = EXPLORATION_WORLD_VERSION
 result["environment_version"] = ENVIRONMENT_VERSION
 result["balance_sha256"] = balance_sha256
 return result

func apply_action(action: Dictionary) -> bool:
 if not rules_valid:return false
 action_start = {"health": health, "hunger": hunger, "thirst": thirst}
 var accepted: bool = super.apply_action(action)
 if accepted and not dead and str(action.verb) == "drink":
  last_event = "在清澈的岸边饮水，水分 +50。"
 elif accepted and not dead and str(action.verb) == "rest":
  last_event = "坐下来稍作休息，恢复体力；食水充足时可在营地恢复生命。"
 action_start.clear()
 return accepted

func _advance(steps: int) -> void:
 # Parent applies its original instant effect before calling this virtual method.
 # Recompute from PRE-action values before any tick/trace, including clamping.
 if last_verb == "drink":
  thirst = minf(100.0, float(action_start.thirst) + float(balance.drink_restore))
 elif last_verb == "rest":
  var can_heal: bool = camp_near() and float(action_start.hunger) >= float(balance.heal_min_hunger) and float(action_start.thirst) >= float(balance.heal_min_thirst)
  health = minf(100.0, float(action_start.health) + (float(balance.camp_rest_health) if can_heal else 0.0))
 for _i in range(steps):
  var health_before_tick: float = health
  tick += 1
  hunger = maxf(0.0, hunger - float(balance.hunger_drain))
  thirst = maxf(0.0, thirst - float(balance.thirst_drain))
  if hunger <= 0.0: health -= float(balance.hunger_damage)
  if thirst <= 0.0: health -= float(balance.thirst_damage)
  if energy <= 0.0: health -= float(balance.exhaustion_damage)
  if is_night() and not camp_near(): health -= float(balance.night_damage)
  for threat in visible_threats():
   if float(threat.distance) < 1.6 and not camp_near(): health -= float(balance.threat_damage)
  health = clampf(health, 0.0, 100.0)
  if health <= 0.0:
   dead = true
   last_event = "旅程结束了。记住这次经历，再开启一个清晨。"
   _record_step(maxf(0.0, health_before_tick - health))
   break
  _record_step(maxf(0.0, health_before_tick - health))
