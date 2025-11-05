import numpy as np
import keras
from keras import ops
from keras import layers
import tensorflow as tf

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_seconds_per_episode = 30
max_episodes = 10

# Define baseline agent
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
    
# Evaluate the baseline agent

sim.load_environment("training_environment")
handle = sim.run_episode(BaselineAgent(), max_seconds_per_episode*60)
sim.print("Running episode for baseline agent.")


while True:
    if handle.is_done():
        break

checkpoint_times, terminated = handle.get_result()
sim.print("Episode for baseline agent complete.")

# Neural Network Agent
class NNAgent:
    import keras
    from keras import ops

    model: keras.Model

    def __init__(self, model: keras.Model):
        self.model = model

    def eval(self, inputs: list[float], state: list[float]):
        tensor_state = ops.convert_to_tensor()
        state = ops.expand_dims(state, 0)

        action_probs, critic_value = model(state)

        model

# Setup actor critic network
num_inputs = 5
num_actions = 2
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

# Train neural network agent
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while episode_count < max_episodes:
    sim.load_environment("training_environment")
    episode_reward = 0
    
    with tf.GradientTape() as tape:
        handle = sim.run_episode(NNAgent(), max_seconds_per_episode*60)

        while True:
            if handle.is_done():
                break