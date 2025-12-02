import numpy as np
import keras
from keras import ops
from keras import layers
import tensorflow as tf
import agents

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_seconds_per_episode = 60
max_steps = max_seconds_per_episode*60
max_episodes = 10
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
    
# Load baseline checkpoint times
baseline_checkpoint_times = np.load('baseline_checkpoint_times.npy')

# Setup actor critic network
num_inputs = 5
num_steering_actions = 3
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action_engine = layers.Dense(num_steering_actions, activation="softmax", name = "steering_out")(common)
critic = layers.Dense(1, activation="linear", name="critic_out")(common)

model = keras.Model(inputs=inputs, outputs=[action_engine, critic])

# Train neural network agent
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
engine_action_probs_history = []
critic_value_history = []
rewards_history = []
episode_count = 0

while episode_count < max_episodes:
    sim.load_environment("training_environment")

    episode_reward = 0
    # Create agent and run episode to get states
    agent = agents.NNSteeringAgent(model, num_steering_actions)
    sim.print(f"Running steering-only training episode {episode_count + 1}.")
    handle = sim.run_episode(agent, max_steps)

    while True:
        if handle.is_done():
            break

    # Upack results
    checkpoint_times, terminated, end_step = handle.get_result()
    state_history = agent.state_history
    action_history = agent.action_history
    states_tf = tf.convert_to_tensor(agent.state_history, dtype=tf.float32)

    sim.print(f" Completed in {end_step} steps")

    with tf.GradientTape() as tape:
        # Use actual model to calculate states, needed to compute gradients
        action_steering, critic_values = model(states_tf, training=True)

        # Extract chosen action probabilities
        steering_indices = tf.constant([a for a in agent.action_history], dtype=tf.int32)
        steering_action_probs_history = tf.gather(action_steering, steering_indices, axis=1, batch_dims=1)
        steering_action_probs_history = tf.math.log(steering_action_probs_history + eps)

        j = 0 # Checkpoint times history

        for step, (state, action) in enumerate(zip(agent.state_history, agent.action_history)):
            # Inputs
            left_distance = state[0]
            forward_distance = state[1]
            right_distance = state[2]
            # State
            speed = state[3]
            steering_angle = state[4]
            # Action
            steering_action = action

            # Compute steering power
            steering_power = 0.0

            if steering_action == 0:
                steering_power = -1.0
            elif steering_action == 1:
                steering_power = 0.0
            else:
                steering_power = 1.0

            # Define tolerance for "centered"
            center_tolerance = 0.1

            # Initialize reward
            reward = 0.0

            # --- Base checkpoint reward ---
            if j < len(checkpoint_times) and step > checkpoint_times[j]:
                j += 1

            if j < len(checkpoint_times):
                baseline_time = baseline_checkpoint_times[j]
                nn_time = checkpoint_times[j]
                # reward = np.maximum(baseline_time - nn_time, 0)

            side_distance_diff = left_distance - right_distance
            side_distance_diff_normalized = np.clip(side_distance_diff / 5.0, -1.0, 1.0)
            min_steering_angle = -30.0*(np.pi/180.0)
            max_steering_angle = 30.0*(np.pi/180.0)
            target_steering_angle = side_distance_diff_normalized * max_steering_angle
            steering_error = target_steering_angle - steering_angle
            steering_angle_diff_normalized = abs(steering_error) / (np.pi / 3)

            # Steering rewards/penalties
            center_tolerance = 0.1

            # Turning
            # if abs(steering_angle_diff) > 1.0*(np.pi/180.0):
            #     if steering_angle < target_steering_angle:
            #         if steering_power == 1:
            #             reward += 0.3*(1 - steering_angle_diff_normalized)*(1 - abs(side_distance_diff_normalized))
            #         elif steering_power == -1:
            #             reward -= 0.3*steering_angle_diff_normalized*(abs(side_distance_diff_normalized))

            #     elif steering_angle > target_steering_angle:
            #         if steering_power == -1:
            #             reward += 0.3*(1 - steering_angle_diff_normalized)*(1 - abs(side_distance_diff_normalized))
            #         elif steering_power == 1:
            #             reward -= 0.3*steering_angle_diff_normalized*(abs(side_distance_diff_normalized))
            # else:
            #     reward += (0.3 if steering_power == 0 else -0.3)

            if abs(steering_error) > np.pi/45.0:
                next_steering_angle = steering_angle + steering_power*np.pi/180.0
                next_steering_error = target_steering_angle - next_steering_angle

                if (abs(steering_error) - abs(next_steering_error)) > 0.0:
                    reward += 0.3*abs(side_distance_diff)
                elif (abs(steering_error) - abs(next_steering_error)) <= 0.0:
                    reward -= 0.3*abs(side_distance_diff)

            else:
                reward += (0.3 if steering_power == 0 else -0.3)

            # Append reward
            if step % 10 == 0:
                sim.print(f"state = ({steering_angle} rad), input = ({left_distance} m, {forward_distance} m, {right_distance} m)")
                sim.print(f"target = ({target_steering_angle} rad), action = ({steering_power}), reward = ({reward})")
            rewards_history.append(reward)
        
        if terminated:
            rewards_history[-1] -= 10

        episode_reward = sum(rewards_history)

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(steering_action_probs_history, critic_values, returns)
        actor_losses = []
        critic_losses = []

        for log_prob_steering, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up receiving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            log_prob = log_prob_steering
            actor_losses.append(-log_prob * diff)  # actor loss
            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(ops.expand_dims(value, 0), ops.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = tf.add_n(actor_losses) + tf.add_n(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_count += 1
        rewards_history = []

        sim.print(f"episode_reward = ({episode_reward})")
    