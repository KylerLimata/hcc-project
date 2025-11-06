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

baseline_checkpoint_times, terminated = handle.get_result()
sim.print("Episode for baseline agent complete.")

# Neural Network Agent
class NNAgent:
    import keras
    from keras import ops

    model: keras.Model

    def __init__(self, model: keras.Model, num_steering_actions: int, num_engine_actions: int):
        self.model = model
        self.num_steering_actions = num_steering_actions
        self.num_engine_actions = num_engine_actions
        self.steering_action_probs_history = []
        self.engine_action_probs_history = []
        self.critic_value_history = []

    def eval(self, inputs: list[float], state: list[float]):
        full_state = inputs + state
        full_state = ops.convert_to_tensor(full_state)
        full_state = ops.expand_dims(state, 0)

        steering_action_probs, engine_action_probs, critic_value = model(full_state)
        steering_action = np.random.choice(self.num_steering_actions, p=np.squeeze(steering_action_probs))
        engine_action = np.random.choice(self.num_engine_actions, p=np.squeeze(engine_action_probs))

        self.steering_action_probs_history.append(ops.log(steering_action_probs[0, steering_action]))
        self.engine_action_probs_history.append(ops.log(engine_action_probs[0, engine_action]))
        self.critic_value_history.append(critic_value[0, 0])

        engine_power = engine_action - 1.0
        steering_direction = steering_action - 1.0

        return [steering_direction, engine_power]

# Setup actor critic network
num_inputs = 5
num_steering_actions = 3
num_engine_actions = 3
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action_steering = layers.Dense(num_steering_actions, activation="softmax")(common)
action_engine = layers.Dense(num_engine_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action_steering, action_engine, critic])

# Train neural network agent
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
steering_action_probs_history = []
engine_action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while episode_count < max_episodes:
    sim.load_environment("training_environment")
    episode_reward = 0
    
    sim.print(f"Running training episode {episode_count + 1}.")

    with tf.GradientTape() as tape:
        # Create agent and run episode
        agent = NNAgent(model, num_engine_actions, num_engine_actions)
        handle = sim.run_episode(agent, max_seconds_per_episode*60)

        while True:
            if handle.is_done():
                break

        # Unpack histories
        critic_value_history = agent.critic_value_history
        steering_action_probs_history = agent.steering_action_probs_history
        engine_action_probs_history = agent.engine_action_probs_history

        # Compute reward for episode
        checkpoint_times, terminated = handle.get_result()

        for i in range(len(checkpoint_times)):
            baseline_time = baseline_checkpoint_times[i]
            nn_time = checkpoint_times[i]

            episode_reward += nn_time - baseline_time
    
    episode_count += 1