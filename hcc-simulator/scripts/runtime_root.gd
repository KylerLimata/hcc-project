extends Node3D

signal complete_episode(checkpoint_times: Array[int], terminated: bool)

var current_environment: Node3D = null
var current_agent: PythonAgent = null
var current_agent_node: AgentVehicleBody = null
var current_step: int = 0
var end_step: int = 0
var checkpoint_times: Array[int] = []
var terminated: bool = false

func load_environment(env_name: String):
	var scene = load("res://scenes/environments/" + env_name + ".tscn").instantiate()
	current_environment = scene
	add_child(scene)

func evaluate_agent(agent: PythonAgent):
	current_agent = agent

func run_episode(agent: PythonAgent, max_steps: int):
	current_agent = agent
	current_step = 0
	end_step = max_steps
	
func _physics_process(_delta: float) -> void:
	# Check if there is a current environment and agent
	if current_environment != null and current_agent != null:
		# Create a new agent if there is none
		if current_agent_node == null:
			var scene = load("res://scenes/agent.tscn").instantiate()
			current_agent_node = scene
			current_environment.add_child(current_agent_node)
			current_environment.connect("checkpoint_activated", on_checkpoint_activated)
			current_agent_node.agent = current_agent
			checkpoint_times = []
			terminated = false
		else:
			# Check if episode has finished
			if current_step >= end_step:
				emit_signal("complete_episode", checkpoint_times, terminated)
				current_environment.remove_child(current_agent_node)
				current_agent_node = null
				current_agent = null
				$VBoxContainer/PanelContainer/MarginContainer/HBoxContainer/RunButton.disabled = false
			else:
				current_step += 1


func _on_script_name_input_output(_filename: Variant) -> void:
	$VBoxContainer/PanelContainer/MarginContainer/HBoxContainer/RunButton.disabled = true

func on_checkpoint_activated(final: bool):
	checkpoint_times.append(current_step)
	
	if final:
		current_step = end_step
