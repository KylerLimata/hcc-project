sim.print("Hello from Python!")

class DummyAgent:
    def __init__(self):
        pass

    def eval_step(self, distances: list[float]):
        return [100.0, 0.0]

sim.load_environment("TestEnvironment")
result = sim.run_episode(DummyAgent(), 100*60)
sim.print("Got the future!")

while True:
    if result.is_done():
        break

data = result.get_result()
sim.print("Finished running!")