@tool
extends Node3D

# EDITOR PROPERTIES
@export_group("Track")
@export_range(5.0, 90.0, 5.0, "suffix:deg") var angle: float = 90.0 : set = set_angle
@export_range(5.0, 15.0, 0.5, "suffix:m") var radius: float = 1.0 : set = set_radius
@export_enum("Left", "Right") var direction: int = 0 : set = set_direction

var queued_ground_array = null
var queued_left_wall_array = null
var queued_right_wall_array = null

func set_angle(val: float):
	angle = val
	
	update_mesh()

func set_radius(val: float):
	radius = val
	
	update_mesh()

func set_direction(val: int):
	direction = val
	
	update_mesh()

func _process(delta: float) -> void:
	if queued_ground_array != null:
		$Ground/CollisionPolygon3D.polygon = queued_ground_array
		queued_ground_array = null
	
	if queued_left_wall_array != null:
		$LeftWall/CollisionPolygon3D.polygon = queued_left_wall_array
		queued_left_wall_array = null
		
	if queued_right_wall_array != null:
		$RightWall/CollisionPolygon3D.polygon = queued_right_wall_array
		queued_right_wall_array = null

func update_mesh():
	var center = null
	
	# A value of 0 indicates a left turn
	if direction == 0:
		center = Vector3(0.0, 0.0, -radius)
	else:
		center = Vector3(0.0, 0.0, radius)
	
	# update the ground
	var ground_inner_points = compute_points(center, radius - 1.5, 0.0)
	var ground_outer_points = compute_points(center, radius + 1.5, 0.0)
	ground_outer_points.reverse()
	
	var ground_polygon = ground_inner_points + ground_outer_points
	queued_ground_array = ground_polygon
	
	# update the left wall
	var left_wall_inner_points = compute_points(center, radius - 1.5, 0.0)
	var left_wall_outer_points = compute_points(center, radius - 1.25, 0.0)
	left_wall_outer_points.reverse()
	queued_left_wall_array = left_wall_inner_points + left_wall_outer_points
	
	# update the right wall
	var right_wall_inner_points = compute_points(center, radius + 1.5, 0.0)
	var right_wall_outer_points = compute_points(center, radius + 1.25, 0.0)
	right_wall_outer_points.reverse()
	queued_right_wall_array = right_wall_inner_points + right_wall_outer_points
	
	print("here!")
	
	pass

func compute_points(center: Vector3, radius_prime: float, phi: float) -> Array[Vector2] :
	var theta: float = 0.0
	var points: Array[Vector2] = []
	
	while (theta <= angle):
		var next = Vector3(0.0, 0.0, radius_prime)
		
		if direction == 0:
			next = Vector3(0.0, 0.0, radius_prime)
			next = next.rotated(Vector3.UP, deg_to_rad(theta + phi))
		else:
			next = Vector3(0.0, 0.0, -radius_prime)
			next = next.rotated(Vector3.UP, deg_to_rad(-(theta + phi)))
		
		next = next + center
		points.append(Vector2(next.x, next.z))
		
		theta = theta + 5.0
	
	return points
