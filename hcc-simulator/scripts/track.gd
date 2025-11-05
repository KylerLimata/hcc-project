@tool extends Node3D
class_name Track

func _physics_process(delta: float) -> void:
	if Engine.is_editor_hint():
		var children: Array[Node] = get_children()
		
		if children.size() > 1:
			for i in range(1, children.size()):
				var current_child: Node3D = children[i - 1]
				var next_child: Node3D = children[i]
				
				var end_transform = current_child.global_transform * current_child.get_end_transform()
				next_child.global_transform = end_transform
