@tool
extends Node3D


# EDITOR PROPERTIES
@export_group("Track")
@export_range(1.0, 100.0, 1.0, "suffix:m") var length: float = 1.0 : set = set_length

func set_length(val: float):
	length = val
	update_mesh()

func update_mesh():
	var ground_shape: BoxShape3D = $Ground/CollisionShape3D.shape
	ground_shape.size.x = length
	
	var ground_mesh: BoxMesh = $Ground/MeshInstance3D.mesh
	ground_mesh.size.x = length
	
	var left_wall_shape: BoxShape3D = $"Left Wall/CollisionShape3D".shape
	left_wall_shape.size.x = length
	
	var left_wall_mesh: BoxMesh = $"Left Wall/MeshInstance3D".mesh
	left_wall_mesh.size.x = length
	
	var right_wall_shape: BoxShape3D = $"Right Wall/CollisionShape3D".shape
	right_wall_shape.size.x = length
	
	var right_wall_mesh: BoxMesh = $"Right Wall/MeshInstance3D".mesh
	right_wall_mesh.size.x = length
	
	$Ground.position.x = length/2
	$"Left Wall".position.x = length/2
	$"Right Wall".position.x = length/2

func get_end_transform():
	return Transform3D(Basis(), Vector3(length, 0, 0))
