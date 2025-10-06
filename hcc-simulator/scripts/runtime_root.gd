extends Node3D

var current_environment: Node3D = null
var current_agent: PythonAgent = null
var current_agent_node: AgentVehicleBody = null

func load_environment(name: String):
	var scene = load("res://scenes/environments/" + name + ".tscn").instantiate()
	current_environment = scene
	add_child(scene)

func evaluate_agent(agent: PythonAgent):
	current_agent = agent
	
func _physics_process(delta: float) -> void:
	if current_agent_node == null and current_environment != null:
		var scene = load("res://scenes/agent.tscn").instantiate()
		current_agent_node = scene
		current_environment.add_child(current_agent_node)
		current_agent_node.attach_agent(current_agent)


func _on_script_name_input_output(filename: Variant) -> void:
	$MarginContainer.hide()
