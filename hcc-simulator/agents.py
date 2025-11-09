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
    
class NNAgent:
    def __init__(self, model, num_steering_actions: int, num_engine_actions: int):
        import tensorflow as tf

        self.model = model
        self.num_steering_actions = num_steering_actions
        self.num_engine_actions = num_engine_actions
        self.steering_action_probs_history = []
        self.engine_action_probs_history = []
        self.critic_value_history = []

        # Persistent buffer for input state (shape [1, 5])
        self.full_state = tf.Variable(
            tf.zeros((1, 5), dtype=tf.float32), trainable=False, name="full_state"
        )

        # Precompile model call with fixed input signature
        self._eval_model = tf.function(
            self.model,
            input_signature=[tf.TensorSpec(shape=(1, 5), dtype=tf.float32)],
        )

    def eval(self, inputs: list[float], state: list[float]):
        import keras
        from keras import ops
        import numpy as np

        full_state_values = np.array(inputs + state, dtype=np.float32).reshape(1, 5)
        self.full_state.assign(full_state_values)

        steering_action_probs, engine_action_probs, critic_value = self._eval_model(self.full_state)
        steering_action = np.random.choice(self.num_steering_actions, p=np.squeeze(steering_action_probs))
        engine_action = np.random.choice(self.num_engine_actions, p=np.squeeze(engine_action_probs))

        self.steering_action_probs_history.append(ops.log(steering_action_probs[0, steering_action]))
        self.engine_action_probs_history.append(ops.log(engine_action_probs[0, engine_action]))
        self.critic_value_history.append(critic_value[0, 0])

        engine_power = engine_action - 1.0
        steering_direction = steering_action - 1.0

        return [steering_direction, engine_power]