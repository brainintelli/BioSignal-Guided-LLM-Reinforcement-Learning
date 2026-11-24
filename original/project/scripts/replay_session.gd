extends RefCounted
## Deterministic historical playback. No policy, network requests or world-rule edits.
const Session = preload("res://scripts/rl_environment_session.gd")
const ACTOR: String = "02111ea2ce34f61647713d348099de4fc577763bc2423e2a1078097e36dff211"
const SOURCE: String = "a072452cc865e4065a6a1b976a8e421c03cd1ffd706c5316cd4ebb5127420f66"
# Explicitly reviewed immutable packages. A self-rewritten sidecar is not trust.
const TRUSTED_PACKAGES: Dictionary = {
	"27dff41b86c3f1c8e028ca549314a452c8224ecb73d6b293cf4c60cbed018719": SOURCE,
	"42973f18712faf6c7873c9c14449a05c335d2b25f93f26cea3849381d3977ef6": "a7fb7fa6b13a3005873ee5ff71e37e137242c87a1fc53911f7d4dcd6225b5832",
}
var engine = Session.new()
var data: Dictionary = {}
var cursor: int = 0
var error: String = ""
var finished: bool = false
var boundary: Dictionary = {}

func load_file(path: String) -> bool:
	var manifest_path: String = path.get_basename() + ".manifest.json"
	if not FileAccess.file_exists(path) or not FileAccess.file_exists(manifest_path):
		return _fail("缺少回放文件或校验清单")
	var manifest: Variant = JSON.parse_string(FileAccess.get_file_as_string(manifest_path))
	if not manifest is Dictionary or str(manifest.get("sha256", "")) != FileAccess.get_sha256(path):
		return _fail("回放文件 SHA256 不匹配，已停止")
	var package_sha: String = str(manifest.sha256)
	if not TRUSTED_PACKAGES.has(package_sha):
		return _fail("回放包不在已审计来源白名单中")
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	if not parsed is Dictionary:
		return _fail("回放文件格式错误")
	data = parsed
	if data.get("format") != "meadow_replay_v1" or data.get("actor_sha256") != ACTOR or data.get("contract") != "open_world_hierarchical_v2":
		return _fail("回放不是已发布的 M6 模型记录")
	if data.get("profile") != "open1" or int(data.get("world_version", 0)) != 2 or data.get("source", {}).get("sha256") != TRUSTED_PACKAGES[package_sha]:
		return _fail("回放环境或来源不匹配")
	if data.source.get("world_sha256") != FileAccess.get_sha256("res://scripts/world_state.gd") or data.source.get("session_sha256") != FileAccess.get_sha256("res://scripts/rl_environment_session.gd"):
		return _fail("世界规则已变化，不能用新规则冒充原始回放")
	if not data.get("records") is Array or data.records.is_empty() or data.records.size() != int(data.get("validated_decisions", -1)):
		return _fail("回放记录不完整")
	return restart()

func restart() -> bool:
	if data.is_empty():
		return _fail("未载入回放")
	error = ""
	cursor = 0
	finished = false
	boundary = {}
	engine = Session.new()
	var reply: Dictionary = engine.handle({"op": "reset", "seed": int(data.seed), "profile": "open1"})
	if not bool(reply.get("ok", false)):
		return _fail("世界重置失败")
	return _match(reply.observation, data.records[0].observation, "初始观察")

func advance() -> bool:
	if finished or not error.is_empty():
		return false
	var record: Dictionary = data.records[cursor]
	if not _match(engine.world.observation(), record.observation, "决策 %d 观察" % cursor):
		return false
	var reply: Dictionary = engine.handle({"op": "step", "action": record.action})
	if not bool(reply.get("ok", false)):
		return _fail("决策 %d 动作不再合法" % cursor)
	if not _match(reply.transition, record.transition, "决策 %d 状态" % cursor) or not _match(reply.terminated, record.terminal, "终止标记"):
		return false
	cursor += 1
	if cursor == data.records.size():
		var limit: int = int(data.display_end_tick)
		for state in reply.transition.trace:
			if int(state.tick) == limit:
				boundary = state.duplicate(true)
		if boundary.is_empty() or not _match(boundary, data.boundary, "记录上限状态"):
			return _fail("缺少可验证的记录终点")
		finished = true
	return true

func display_tick() -> int:
	return int(boundary.tick) if finished else engine.world.tick

func apply_presentation_boundary() -> void:
	# The final full transition was verified above. Render only its recorded time cap.
	# Never execute another action from this presentation-only state; restart rebuilds it.
	if not finished or boundary.is_empty():
		return
	engine.world.tick = int(boundary.tick)
	for key in ["health", "hunger", "thirst", "energy"]:
		engine.world.set(key, float(boundary[key]))
	engine.world.dead = bool(boundary.dead)
	engine.world.player = Vector2i(int(boundary.player.x), int(boundary.player.y))
	engine.world.inventory = boundary.inventory.duplicate()
	engine.world.camp = boundary.camp.duplicate()

func _fail(message: String) -> bool:
	error = message
	return false

func _match(actual: Variant, expected: Variant, location: String) -> bool:
	if (actual is int or actual is float) and (expected is int or expected is float):
		if absf(float(actual) - float(expected)) > 0.000001:
			return _fail(location + "：数值不一致")
		return true
	if actual is Dictionary and expected is Dictionary:
		if actual.size() != expected.size():
			return _fail(location + "：字段数量不一致")
		for key in expected:
			if not actual.has(key) or not _match(actual[key], expected[key], location + "." + str(key)):
				if error.is_empty():
					_fail(location + "：缺少字段 " + str(key))
				return false
		return true
	if actual is Array and expected is Array:
		if actual.size() != expected.size():
			return _fail(location + "：列表长度不一致")
		for i in range(expected.size()):
			if not _match(actual[i], expected[i], location + "[%d]" % i):
				return false
		return true
	if actual != expected:
		return _fail(location + "：内容不一致")
	return true
