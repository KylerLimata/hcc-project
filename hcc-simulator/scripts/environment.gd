extends Node3D
class_name TrainingEnvironment

signal checkpoint_activated(final)

func on_checkpoint_activated(final: bool):
	emit_signal("checkpoint_activated", final)
