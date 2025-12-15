extends Node3D

signal complete_episode(checkpoint_times: Array[int], terminated: bool, end_step: int)

var current_environment: Node3D = null
var current_agent: PythonAgent = null
var current_agent_node: AgentVehicleBody = null
var current_step: int = 0
var end_step: int = 0
var checkpoint_times: Array[int] = []
var terminated: bool = false
var finished: bool = false
var succeeded: bool = false

func load_environment(env_name: String):
	if current_environment != null:
		remove_child(current_environment)
		current_environment.queue_free()
	
	var scene = load("res://scenes/environments/" + env_name + ".tscn").instantiate()
	current_environment = scene
	current_environment.connect("checkpoint_activated", on_checkpoint_activated)
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
			current_agent_node.agent = current_agent
			current_agent_node.on_collide.connect(on_agent_collide)
			checkpoint_times = []
			terminated = false
			finished = false
		else:
			# Check if episode has finished
			if current_step >= end_step:
				finished = true
			else:
				current_step += 1
		
		if finished:
			emit_signal("complete_episode", checkpoint_times, succeeded, terminated, current_step)
			current_environment.remove_child(current_agent_node)
			current_agent_node.queue_free()
			current_agent_node = null
			current_agent = null
			$VBoxContainer/PanelContainer/MarginContainer/HBoxContainer/RunButton.disabled = false


func _on_script_name_input_output(_filename: Variant) -> void:
	$VBoxContainer/PanelContainer/MarginContainer/HBoxContainer/RunButton.disabled = true

func on_checkpoint_activated(final: bool):
	checkpoint_times.append(current_step)
	
	if final:
		finished = true
		succeeded = true

func on_agent_collide():
	terminated = true
	finished = true
