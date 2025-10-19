extends Node3D

signal checkpoint_activated(final)

func on_checkpoint_activated(final: bool):
	emit_signal("checkpoint_activated", final)
