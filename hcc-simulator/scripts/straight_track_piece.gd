@tool
extends Node3D

@export_group("Track")
@export_range(1.0, 100.0, 1.0, "suffix:m") var length: float = 1.0 : set = set_length

var queued_ground_array = null
var queued_left_wall_array = null
var queued_right_wall_array = null

func set_length(val: float):
	length = val
	update_mesh()

func _process(delta: float) -> void:
	if queued_ground_array != null:
		$Ground/CollisionPolygon3D.polygon = queued_ground_array
		$Ground/CSGPolygon3D.polygon = queued_ground_array
		queued_ground_array = null
	
	if queued_left_wall_array != null:
		$LeftWall/CollisionPolygon3D.polygon = queued_left_wall_array
		$LeftWall/CSGPolygon3D.polygon = queued_left_wall_array
		queued_left_wall_array = null
		
	if queued_right_wall_array != null:
		$RightWall/CollisionPolygon3D.polygon = queued_right_wall_array
		$RightWall/CSGPolygon3D.polygon = queued_right_wall_array
		queued_right_wall_array = null

func update_mesh():
	queued_ground_array = [
		Vector2(-2.0, 0.0),
		Vector2(-2.0, length),
		Vector2(2.0, length),
		Vector2(2.0, 0.0)
	]
	queued_left_wall_array = [
		Vector2(-2.0, 0.0),
		Vector2(-2.0, length),
		Vector2(-1.75, length),
		Vector2(-1.75, 0.0)
	]
	queued_right_wall_array = [
		Vector2(2.0, 0.0),
		Vector2(2.0, length),
		Vector2(1.75, length),
		Vector2(1.75, 0.0)
	]

func get_end_transform():
	return Transform3D(Basis(), Vector3(0, 0, length))
