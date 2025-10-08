sim.print("Hello from Python!")

class DummyAgent:
    def __init__(self):
        pass

    def eval_step(self, f: float):
        return [100.0, 0.0]

sim.load_environment("TestEnvironment")
sim.evaluate_agent(DummyAgent())