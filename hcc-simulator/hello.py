class BaselineAgent:
    def __init__(self):
        pass
    
    def eval(self, inputs: list[float], state: list[float]):
        import math

        # Unpack input vec
        left_distance = inputs[0]
        forward_distance = inputs[1]
        right_distance = inputs[2]
        # Unpack state vec
        speed = state[0]
        steering_angle = state[1]

        target_speed = 20*(forward_distance)
        speed_diff = speed - target_speed
        engine_force = 0.0

        if speed_diff < 1.0:
            engine_force = 1.0
        elif speed_diff > 1.0:
            engine_force = -1.0

        side_distance_diff = left_distance - right_distance
        steering_direction = 0.0
        side_distance_diff_normalized = max(-1.0, min(1.0, side_distance_diff / 5.0))
        min_steering_angle = -30.0*(math.pi/180.0)
        max_steering_angle = 30.0*(math.pi/180.0)
        target_steering_angle = min_steering_angle + (side_distance_diff_normalized + 1.0) * ((max_steering_angle - min_steering_angle) / 2.0)
        steering_angle_diff = steering_angle - target_steering_angle

        if steering_angle_diff > 1.0*(math.pi/180.0):
            steering_direction = -1.0
        elif steering_angle_diff < -1.0*(math.pi/180.0):
            steering_direction = 1.0
            
        return [engine_force, steering_direction]

sim.load_environment("training_environment")
handle = sim.run_episode(BaselineAgent(), 100*60)
sim.print("Running episode for baseline agent.")


while True:
    if handle.is_done():
        break

checkpoint_times, terminated = handle.get_result()
sim.print("Episode for baseline agent complete.")