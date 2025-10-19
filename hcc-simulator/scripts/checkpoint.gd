@tool extends Node3D

signal checkpoint_activated(final)

@export var final: bool = false
var triggered = false

func get_end_transform():
	return Transform3D()

func _on_area_3d_body_entered(body: Node3D) -> void:
	if not triggered:
		triggered = true
		emit_signal("checkpoint_activated", final)
