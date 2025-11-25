import agents
import numpy as np

# Evaluate the baseline agent
sim.load_environment("training_environment_new")
agent = agents.DebugAgent()
handle = sim.run_episode(agent, 30*60)
sim.print("Running debug.")

while True:
    if handle.is_done():
        break

baseline_checkpoint_times, _, end_step = handle.get_result()
sim.print(f"Finished running debug agent in {end_step} timesteps ({end_step/60.0} s).")
sim.print(f"States = {agent.states}")

