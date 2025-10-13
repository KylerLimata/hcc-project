extends Node3D

signal return_agent(agent: PythonAgent)

var current_environment: Node3D = null
var current_agent: PythonAgent = null
var current_agent_node: AgentVehicleBody = null
var steps_remaining: int = 0

func load_environment(name: String):
	var scene = load("res://scenes/environments/" + name + ".tscn").instantiate()
	current_environment = scene
	add_child(scene)

func evaluate_agent(agent: PythonAgent):
	current_agent = agent

func run_episode(agent: PythonAgent, max_steps: int):
	current_agent = agent
	steps_remaining = max_steps
	
func _physics_process(delta: float) -> void:
	# Check if there is a current environment and agent
	if current_environment != null and current_agent != null:
		
		# Create a new agent if there is none
		if current_agent_node == null:
			var scene = load("res://scenes/agent.tscn").instantiate()
			current_agent_node = scene
			current_environment.add_child(current_agent_node)
			current_agent_node.agent = current_agent
		else:
			
			# Check if episode has finished
			if steps_remaining <= 0:
				emit_signal("return_agent", current_agent_node.agent)
				current_environment.remove_child(current_agent_node)
				
			else:
				steps_remaining -= 1


func _on_script_name_input_output(filename: Variant) -> void:
	$MarginContainer.hide()
