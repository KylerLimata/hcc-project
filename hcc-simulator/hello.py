sim.print("Hello from Python!")

class DummyAgent:
    def __init__(self):
        pass

    def eval_step(self, distances: list[float]):
        return [100.0, 0.0]

sim.load_environment("TestEnvironment")
handle = sim.run_episode(DummyAgent(), 5*60)
sim.print("Got the handle!")

while True:
    if handle.is_done():
        break

checkpoint_times, terminated = handle.get_result()
sim.print("Finished running!")