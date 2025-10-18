extends Node3D

signal checkpoint_activated()

func on_checkpoint_activated():
	emit_signal("checkpoint_activated")
