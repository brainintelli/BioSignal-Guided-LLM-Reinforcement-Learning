extends RefCounted
## Presentation-only stride state driven by physical displacement, never world goals.

enum { DOWN, LEFT, RIGHT, UP }
const FRAME_DISTANCE: float = 13.0
const TELEPORT_DISTANCE: float = 104.0
const IDLE_GRACE: float = 0.045
var facing: int = DOWN
var frame: int = 0
var moving: bool = false
var last_position: Vector2 = Vector2.ZERO
var travelled: float = 0.0
var stationary_time: float = 0.0
var initialized: bool = false

func reset(position: Vector2) -> void:
	last_position = position
	travelled = 0.0
	stationary_time = 0.0
	facing = DOWN
	frame = 0
	moving = false
	initialized = true

func observe(position: Vector2, delta: float) -> void:
	if not initialized:
		reset(position)
		return
	var displacement: Vector2 = position - last_position
	last_position = position
	var distance: float = displacement.length()
	if distance > TELEPORT_DISTANCE:
		reset(position)
		return
	if distance > 0.01:
		if absf(displacement.x) > absf(displacement.y):
			facing = RIGHT if displacement.x > 0.0 else LEFT
		else:
			facing = DOWN if displacement.y > 0.0 else UP
		travelled += distance
		stationary_time = 0.0
		moving = true
		frame = int(travelled / FRAME_DISTANCE) % 4
	else:
		stationary_time += delta
		# Render frames between physics ticks retain their pose without advancing it.
		if stationary_time >= IDLE_GRACE:
			moving = false
			frame = 0
