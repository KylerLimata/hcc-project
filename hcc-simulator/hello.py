sim.print("Hello from Python!")

class DummyAgent:
    def __init__(self):
        pass

    def eval_step(self, distances: list[float]):
        return [100.0, 0.0]

sim.load_environment("TestEnvironment")
sim.run_episode(DummyAgent(), 100)
sim.print("Finished running!")