sim.print("Hello from Python!")

class DummyAgent:
    def __init__(self):
        pass
    
    def eval(self, inputs: list[float], state: list[float]):
        # Code for computing engine force and state using inputs

        return [1.0, 0.0]

sim.load_environment("TestEnvironment")
handle = sim.run_episode(DummyAgent(), 5*60)
sim.print("Got the handle!")

while True:
    if handle.is_done():
        break

checkpoint_times, terminated = handle.get_result()
sim.print("Finished running!")