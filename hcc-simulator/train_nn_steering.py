import numpy as np
import keras
from keras import ops
from keras import layers
import tensorflow as tf
import agents

# Configuration parameters for the whole setup
seed = 42
gamma = 0.8  # Discount factor for past rewards
max_seconds_per_episode = 60
max_steps = max_seconds_per_episode*60
max_episodes = 1000
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

# Entropy parameters
stagnation_count = 0
last_end_step = 0
base_entropy_coef = 0.1      # increase if too weak later
    
# Load baseline checkpoint times
baseline_checkpoint_times = np.load('baseline_checkpoint_times.npy')

# Setup actor critic network
num_inputs = 7
num_steering_actions = 3
num_hidden = 128

initializer = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action_engine = layers.Dense(num_steering_actions, activation="softmax", name = "steering_out",
                             kernel_initializer=initializer, bias_initializer='zeros')(common)
critic = layers.Dense(1, activation="linear", name="critic_out")(common)

model = keras.Model(inputs=inputs, outputs=[action_engine, critic])

# Train neural network agent
optimizer = keras.optimizers.Adam(learning_rate=3e-3)
huber_loss = keras.losses.Huber()
engine_action_probs_history = []
critic_value_history = []
rewards_history = []
episode_count = 0

while episode_count < max_episodes:
    sim.load_environment("training_environment_new")

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

    sim.print(f" Completed in {end_step} steps")

    with tf.GradientTape() as tape:
        states_tf = tf.convert_to_tensor(agent.state_history, dtype=tf.float32)
        # Use actual model to calculate states, needed to compute gradients
        action_probs, critic_values = model(states_tf, training=False)

        # sim.print(f"action_probs = {action_probs.numpy()}")

        # Extract chosen action probabilities
        steering_indices = tf.constant(agent.action_history, dtype=tf.int32)
        steering_action_probs_history = tf.gather(action_probs, steering_indices, axis=1, batch_dims=1)
        steering_action_probs_history = tf.math.log(tf.clip_by_value(steering_action_probs_history, eps, 1.0))

        j = 0 # Checkpoint times history

        for step, (state, action) in enumerate(zip(agent.state_history, agent.action_history)):
            # Inputs
            left_dist = state[0]
            left_forward_dist = state[1]
            forward_dist = state[2]
            right_forward_dist = state[3]
            right_dist = state[4]
            # State
            speed = state[5]
            steering_angle = state[6]

            # Track previous side error for progress reward
            if step == 0:
                prev_side_error = abs(left_dist - right_dist)
            
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
                # reward += 0.1*np.maximum(baseline_time - nn_time, 0)

            side_dist_diff = left_dist - right_dist
            side_dist_diff_norm = max(-1.0, min(1.0, (left_dist - right_dist)/10.0))
            forward_dist_diff_norm = max(-1.0, min(1.0, (left_forward_dist - right_forward_dist)/10.0))
            min_steering = -30.0*(np.pi/180.0)
            max_steering = 30.0*(np.pi/180.0)
            target_steering_angle = 0.0

            if abs(side_dist_diff_norm) > center_tolerance:
                alpha = min_steering + (side_dist_diff_norm + 1.0) * ((max_steering - min_steering) / 2.0)
                beta = min_steering + (forward_dist_diff_norm + 1.0) * ((max_steering - min_steering) / 2.0)
                target_steering_angle = 0.75*alpha + 0.25*beta

            steering_err = steering_angle - target_steering_angle
            steering_err_norm = abs(steering_err) / (np.pi / 3)

            # Steering rewards/penalties
            side_error_factor = abs(side_dist_diff)
            speed_factor = max(0.0, 0.1*speed)

            if steering_power == -1:
                reward += 10.0*steering_err*(side_error_factor + speed_factor)
            elif steering_power == 0:
                reward += (20.0 if abs(steering_err) < np.pi/90 else -20.0)
            elif steering_power == 1:
                reward += -10.0*steering_err*(side_error_factor + speed_factor)
            else:
                sim.print("Invalid steering power!")
            
            # side_progress = (prev_side_error - side_error)
            # side_progress = float(np.clip(side_progress, -0.05, 0.05))
            # reward += 0.2 * side_progress
            # prev_side_error = side_error

            # Debugging
            # if step % 10 == 0:
            #     sim.print(f"state = ({steering_angle:.2f} rad), input = ({left_dist:.2f} m, {left_forward_dist:.2f} m, {forward_dist:.2f} m, {right_forward_dist:.2f} m, {right_dist:.2f} m)")
            #     sim.print(f"target = ({target_steering_angle:.2f} rad), action = ({steering_power:.2f}), reward = ({reward:.2f})")

            # Append reward
            rewards_history.append(reward)
        
        if terminated:
            crash_penalty = 50 * (1.0 - (end_step / max_steps))
            rewards_history[-1] -= crash_penalty

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

        # Normalize Returns
        returns = np.array(returns, dtype=np.float32)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns_tf = tf.convert_to_tensor(returns, dtype=tf.float32)

        # Normalize Diff
        values_tf = tf.squeeze(critic_values)
        diffs = returns_tf - values_tf      # shape (T,)
        diffs = (diffs - tf.reduce_mean(diffs)) / (tf.math.reduce_std(diffs) + eps)
        diffs = tf.clip_by_value(diffs, -2.0, 2.0)

        # Compute loss
        entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + eps), axis=1)
        entropy_loss = tf.reduce_mean(entropy)
        entropy_coef = base_entropy_coef

        # Actor + Critic losses
        actor_losses = -tf.reduce_mean(steering_action_probs_history * diffs)
        critic_losses = tf.reduce_mean(huber_loss(returns_tf, values_tf))
        loss_value = actor_losses + critic_losses - entropy_coef*entropy_loss

        # Backpropagation
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_count += 1
        rewards_history = []

        sim.print(f"episode_reward = ({episode_reward:.2f})")
    