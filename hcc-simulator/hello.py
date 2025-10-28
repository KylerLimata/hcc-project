class BaselineAgent:
    def __init__(self):
        pass
    
    def eval(self, inputs: list[float], state: list[float]):
        # Unpack input vec
        left_distance = inputs[0]
        forward_distance = inputs[1]
        right_distance = inputs[2]
        # Unpack state vec
        speed = state[0]
        steering_angle = state[1]

        target_speed = 20*(forward_distance)
        speed_diff = target_speed - speed
        engine_force = 0.0

        if speed_diff > 1.0:
            engine_force = 1.0
        elif speed_diff < 1.0:
            engine_force = -1.0

        side_distance_diff = left_distance - right_distance
        steering_direction = 0.0

        if side_distance_diff > 1.0:
            steering_direction = 1.0
        elif side_distance_diff < -1.0:
            steering_direction = -1.0

        return [engine_force, steering_direction]

sim.load_environment("TestEnvironment")
handle = sim.run_episode(BaselineAgent(), 20*60)
sim.print("Running episode for baseline agent.")

while True:
    if handle.is_done():
        break

checkpoint_times, terminated = handle.get_result()
sim.print("Episode for baseline agent complete.")