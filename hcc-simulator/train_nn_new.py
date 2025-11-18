import numpy as np
import keras
from keras import ops
from keras import layers
import tensorflow as tf
import agents

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_seconds_per_episode = 30
max_episodes = 15
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
    
# Load baseline checkpoint times
baseline_checkpoint_times = np.load('baseline_checkpoint_times.npy')

# Setup actor critic network
num_inputs = 5
num_steering_actions = 3
num_engine_actions = 3
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action_steering = layers.Dense(num_steering_actions, activation="softmax", name = "steering_out")(common)
action_engine = layers.Dense(num_engine_actions, activation="softmax", name = "engine_out")(common)
critic = layers.Dense(1, activation="linear", name="critic_out")(common)

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

sim.load_environment("training_environment")

while episode_count < max_episodes:
    episode_reward = 0
    # Create agent and run episode to get states
    agent = agents.NewFastNNAgent(model, num_steering_actions, num_engine_actions)
    sim.print(f"Running training episode {episode_count + 1}.")
    handle = sim.run_episode(agent, max_seconds_per_episode*60)

    while True:
        if handle.is_done():
            break

    # Upack results
    checkpoint_times, terminated, end_step = handle.get_result()
    state_history = agent.state_history
    action_history = agent.action_history
    states_tf = tf.convert_to_tensor(agent.state_history, dtype=tf.float32)
    rewards_history = [0.0]*end_step

    with tf.GradientTape() as tape:
        # Use actual model to calculate states, needed to compute gradients
        action_steering, action_engine, critic_values = model(states_tf, training=True)

        # Extract chosen action probabilities
        steering_indices = tf.constant([a[0] for a in agent.action_history], dtype=tf.int32)
        engine_indices   = tf.constant([a[1] for a in agent.action_history], dtype=tf.int32)

        steering_action_probs_history = tf.gather(action_steering, steering_indices, axis=1, batch_dims=1)
        engine_action_probs_history   = tf.gather(action_engine, engine_indices, axis=1, batch_dims=1)

        # Fill out rewards history
        # - Reward the agent for getting to a checkpoint faster than baseline
        # for i in range(len(checkpoint_times)):
        #     baseline_time = baseline_checkpoint_times[i]
        #     nn_time = checkpoint_times[i]
        #     reward = np.max([nn_time - baseline_time, 0])
        #     rewards_history[nn_time] = reward

        j = 0 # Checkpoint times history
        
        sim.print("Computing rewards!")

        # - Rewards based on state
        for i in range(len(agent.state_history) - 1):
            state = agent.state_history[i]
            next_state = agent.state_history[i + 1]
            # Inputs
            left_distance = state[0]
            next_left_distance = next_state[0]
            forward_distance = state[1]
            next_forward_distance = next_state[1]
            right_distance = state[2]
            next_right_distance = next_state[2]
            # State
            speed = state[3]
            next_speed = next_state[3]
            steering_angle = state[4]
            next_steering_angle = next_state[4]

            reward = 0.0

            if j < len(checkpoint_times) and i > checkpoint_times[j]:
                j += 1

            # Calculate base reward from checkpoint time
            if j < len(checkpoint_times):
                baseline_time = baseline_checkpoint_times[j]
                nn_time = checkpoint_times[j]
                reward = np.max([nn_time - baseline_time, 0])

            # Compute rewards and penalities
            delta_speed = next_speed - speed
            delta_steering_angle = next_steering_angle - steering_angle
            side_distance_diff = left_distance - right_distance
            next_side_distance_diff = next_left_distance - next_right_distance
            center_tolerance = 0.1

            # Car should probably be turning
            if abs(side_distance_diff) > center_tolerance and forward_distance < 5.0:
                # Reward based on change in forward distance
                if speed > 0.0:
                    reward += 0.05 * (next_forward_distance - forward_distance)

                # Penalize steering right when close to right wall
                if side_distance_diff > center_tolerance and delta_steering_angle >= 0.0:
                    reward -= 0.5*(5 - right_distance)
                # Penalize steering left when close to left wall
                elif side_distance_diff < -center_tolerance and delta_steering_angle <= 0.0:
                    reward -= 0.5*(5 - left_distance)
            # Car should probably be going straight
            else:
                # Small reward based on speed
                reward += 0.05 * speed

                # Penalize unnecessary steering
                if abs(delta_steering_angle) > 0.1:
                    reward -= 1.0

            rewards_history[i] = reward
        
        for r in rewards_history:
            episode_reward += r

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
    
        episode_count += 1

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        sim.print(f"len(rewards_history) = {len(rewards_history)}")
        sim.print(f"len(returns) = {len(returns)}")
    
        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(steering_action_probs_history, engine_action_probs_history, critic_values, returns)
        actor_losses = []
        critic_losses = []

        for log_prob_steering, log_prob_engine, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up receiving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            log_prob = log_prob_steering + log_prob_engine
            actor_losses.append(-log_prob * diff)  # actor loss
            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(ops.expand_dims(value, 0), ops.expand_dims(ret, 0))
            )
        
        # Backpropagation
        loss_value = tf.add_n(actor_losses) + tf.add_n(critic_losses)
        sim.print(" Computing gradients...")
        grads = tape.gradient(loss_value, model.trainable_variables)
        sim.print(" Applying gradients...")
        none_count = sum(1 for g in grads if g is None)
        sim.print(f"DEBUG: total grads = {len(grads)}, None grads = {none_count}")

        for i, (g, v) in enumerate(zip(grads, model.trainable_variables)):
            name = v.name
            if g is None:
                sim.print(f"[{i}] VAR {name}: grad = None, shape={v.shape}, dtype={v.dtype}")
            else:
                try:
                    # eager numeric check
                    s = float(tf.reduce_sum(tf.abs(g)).numpy())
                    mn = float(tf.reduce_min(g).numpy())
                    mx = float(tf.reduce_max(g).numpy())
                    sim.print(f"[{i}] VAR {name}: grad shape={g.shape}, dtype={g.dtype}, sum_abs={s:.6g}, min={mn:.6g}, max={mx:.6g}")
                except Exception as e:
                    sim.print(f"[{i}] VAR {name}: grad present but numeric read failed: {e}")
        
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        sim.print(f" gradients applied.")

        sim.print(f" reward: {episode_reward}")