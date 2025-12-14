import agents
import numpy as np

# Evaluate the baseline agent
sim.load_environment("training_environment_new")
handle = sim.run_episode(agents.BaselineAgent(breaking=False), 60*60)
sim.print("Running baseline agent.")

while True:
    if handle.is_done():
        break

baseline_checkpoint_times, _, end_step = handle.get_result()
sim.print(f"Finished running baseline agent in {end_step} timesteps ({end_step/60.0} s). Saving checkpoint times.")

np.save('baseline_checkpoint_times.npy', np.array(baseline_checkpoint_times))
sim.print("Checkpoint times saved.")

