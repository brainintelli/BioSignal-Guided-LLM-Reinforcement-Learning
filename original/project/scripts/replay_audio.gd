extends Node
## Existing 8-bit assets only. One music voice and one replaceable effect voice.
var music: AudioStreamPlayer
var effects: AudioStreamPlayer
var muted: bool = false
var active: bool = false
var night: bool = false
var tracks: Dictionary = {}
var sounds: Dictionary = {}

func _ready() -> void:
	music = AudioStreamPlayer.new()
	music.volume_db = -16.0
	music.max_polyphony = 1
	add_child(music)
	effects = AudioStreamPlayer.new()
	effects.volume_db = -13.0
	effects.max_polyphony = 1
	add_child(effects)
	for key in ["meadow_loop", "night_loop", "gather", "eat", "drink", "build"]:
		var path: String = "res://assets/audio/%s.wav" % key
		if ResourceLoader.exists(path):
			if key.ends_with("_loop"):
				tracks[key] = load(path)
			else:
				sounds[key] = load(path)
	music.finished.connect(func() -> void:
		if active and not muted and is_inside_tree():
			music.play())

func sync(is_playing: bool, is_night: bool) -> void:
	active = is_playing and not muted
	night = is_night
	if not active:
		stop()
		return
	var key: String = "night_loop" if night else "meadow_loop"
	var track: AudioStream = tracks.get(key)
	if track == null:
		return
	if music.stream != track:
		music.stop()
		music.stream = track
	if not music.playing:
		music.play()

func play_effect(verb: String) -> void:
	if not active or muted:
		return
	var key: String = "build" if verb == "build_camp" else verb
	if not sounds.has(key):
		return
	# Replace rather than stack, even when recorded actions play at 8x.
	effects.stop()
	effects.stream = sounds[key]
	effects.play()

func stop() -> void:
	active = false
	if music != null:
		music.stop()
		music.stream = null
	if effects != null:
		effects.stop()
		effects.stream = null

func _exit_tree() -> void:
	stop()
	tracks.clear()
	sounds.clear()
