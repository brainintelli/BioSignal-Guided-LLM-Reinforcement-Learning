extends SceneTree
## Persistent headless stdin/stdout JSONL bridge. Requests use ASCII JSON encoding.
## Launch: godot --headless --path PROJECT --script godot_tests/m8b_environment.gd
## Output prefix separates protocol replies from engine banners and diagnostics.

const Session = preload("res://scripts/m8b_environment_session.gd")
const PREFIX: String = "MM_RL_JSON="
const MAX_LINE_LENGTH: int = 65536

func _initialize() -> void:
	var session = Session.new()
	var pending: String = ""
	var overflow: bool = false
	while true:
		# Pipe reads otherwise wait for the default buffer size, deadlocking RPC.
		# String reads strip line endings on Windows; raw bytes preserve framing.
		var data: PackedByteArray = OS.read_buffer_from_stdin(1)
		if data.is_empty():
			break
		var character: String = char(data[0])
		if character != "\n":
			if pending.length() < MAX_LINE_LENGTH:
				pending += character
			else:
				overflow = true
			continue
		if overflow:
			print(PREFIX + JSON.stringify({"id": null, "ok": false, "error": "request_too_large"}))
			pending = ""
			overflow = false
			continue
		var parser := JSON.new()
		var response: Dictionary
		if parser.parse(pending) != OK or not parser.data is Dictionary:
			response = {"id": null, "ok": false, "error": "invalid_json_object"}
		else:
			response = session.handle(parser.data)
		pending = ""
		print(PREFIX + JSON.stringify(response))
		if bool(response.get("closed", false)):
			break
	quit(0)
