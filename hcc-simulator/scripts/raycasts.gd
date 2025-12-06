extends Node3D

signal update_raycast_distances(distances: Array[float])

var distances: Array[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
var rays: Array[RayCast3D] = []

func _ready() -> void:
	rays = [$LeftRaycast, $ForwardLeftRaycast, $ForwardRaycast, $ForwardRightRaycast, $RightRaycast]

func _physics_process(delta: float) -> void:
	for i in range(rays.size()):
		var ray: RayCast3D = rays[i]
		
		if ray.is_colliding():
			distances[i] = global_position.distance_to(ray.get_collision_point())
		else:
			distances[i] = 10.0
		
		pass
	
	emit_signal("update_raycast_distances", distances)
