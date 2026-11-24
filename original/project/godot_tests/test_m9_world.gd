extends SceneTree
const NewWorld = preload("res://scripts/m9_world.gd")
const OldWorld = preload("res://scripts/survival_world.gd")
var failures: int = 0
func check(ok: bool, label: String) -> void:
	if not ok:
		failures += 1
		print("FAIL " + label)
func _initialize() -> void:
	var old = OldWorld.new()
	old.reset(2304, "open4")
	var finite = NewWorld.new()
	finite.configure("finite", 1)
	finite.reset(2304, "m9")
	for y in range(96):
		for x in range(96):
			check(old.terrain_at(Vector2i(x,y)) == finite.terrain_at(Vector2i(x,y)), "origin terrain preserved")
	check(old.allowed_actions() == finite.allowed_actions(), "finite legal actions preserved")
	for i in range(12):
		old.apply_action({"verb":"wait"})
		finite.apply_action({"verb":"wait"})
	check(old.health == finite.health and old.hunger == finite.hunger and old.thirst == finite.thirst, "finite vitals preserved")
	var a = NewWorld.new()
	a.configure("infinite_freeze", 1)
	a.reset(2304, "m9")
	var b = NewWorld.new()
	b.configure("infinite_freeze", 1)
	b.reset(2304, "m9")
	# Generation is independent of entry order, including negative chunk coordinates.
	a.ensure_chunk(Vector2i(-2,0)); a.ensure_chunk(Vector2i(4,-1))
	b.ensure_chunk(Vector2i(4,-1)); b.ensure_chunk(Vector2i(-2,0))
	check(a.chunks[Vector2i(-2,0)] == b.chunks[Vector2i(-2,0)], "negative generation order independent")
	check(a.chunks[Vector2i(4,-1)] == b.chunks[Vector2i(4,-1)], "positive generation order independent")
	check(a.chunk_of(Vector2i(-1,-65)) == Vector2i(-1,-2), "negative floor division")
	# Pick a traversable boundary tile; entering does not reset vitals or tick.
	a.player = Vector2i(94,48); a.activate_region()
	var crossed: bool = false
	for y in range(10,85):
		if a.is_walkable(Vector2i(94,y)) and a.is_walkable(Vector2i(95,y)):
			a.player = Vector2i(94,y); a._reveal()
			crossed = a.apply_action({"verb":"move","target_id":a.tile_id(Vector2i(95,y))})
			break
	check(crossed and a.player.x == 95 and a.tick == 1, "old border is traversable and costs one tick")
	check(a.hunger < 82.0 and a.thirst < 78.0, "crossing consumes needs")
	# Frozen chunks retain depleted resources and their local time through re-entry.
	a.player = Vector2i(48,48); a.activate_region()
	var origin: Dictionary = a.chunks[Vector2i.ZERO]
	var resource: Dictionary = origin.resources[0]
	resource.ready_local = int(origin.local_tick) + 90
	var deadline: int = int(resource.ready_local)
	var local_before: int = int(origin.local_tick)
	a.player = Vector2i(256,256); a.activate_region()
	a.apply_action({"verb":"wait"}); a.apply_action({"verb":"wait"})
	check(int(origin.local_tick) == local_before, "inactive region clock frozen")
	a.player = Vector2i(48,48); a.activate_region()
	check(int(resource.ready_local) == deadline, "reentry does not respawn resources")
	a.apply_action({"verb":"wait"})
	check(int(origin.local_tick) == local_before+1, "reactivated region advances")
	# The negative half-plane must not inherit the old x>=0 camp-placement sentinel.
	a.player = Vector2i(-50,-50); a.activate_region()
	a.inventory.wood=10; a.inventory.stone=10; a.camp.exists=false
	var site_found: bool = false
	for y in range(-60,-40):
		for x in range(-60,-40):
			a.player=Vector2i(x,y)
			var exits: int = 0
			for direction in a.DIRECTIONS:
				if a.is_walkable(a.player+direction) and a.resource_at(a.player+direction).is_empty(): exits+=1
			if a.is_walkable(a.player) and exits>=2:
				site_found=true
				check(a.allowed_actions().has({"verb":"build_camp"}), "negative coordinates allow legal camp construction")
				break
		if site_found: break
	check(site_found,"negative camp test fixture")
	var macro = NewWorld.new()
	macro.configure("finite",4); macro.reset(2304,"m9")
	var nav: Dictionary = {}
	for action in macro.allowed_actions():
		if action.verb == "navigate": nav = action; break
	check(not nav.is_empty(), "navigation fixture exists")
	macro.apply_action(nav)
	check(macro.tick > 1 and macro.tick <= 4, "macro advances all physical ticks")
	check(macro.last_step_trace.size() == macro.tick, "macro preserves per-tick trace")
	check(macro.inventory.berries == 2, "navigation does not gather or eat")
	check(is_equal_approx(macro.hunger,82.0-0.2*macro.tick), "macro charges hunger per tick")
	print("M9_WORLD_RESULT failures=" + str(failures))
	quit(1 if failures else 0)
