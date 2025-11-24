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
max_episodes = 10
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
    
# Load baseline checkpoint times
baseline_checkpoint_times = np.load('baseline_checkpoint_times.npy')

# Setup actor critic network
num_inputs = 5
num_steering_actions = 3
num_engine_actions = 3
num_hidden = 512

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

while episode_count < max_episodes:
    sim.load_environment("training_environment")

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
        for i in range(len(checkpoint_times)):
            baseline_time = baseline_checkpoint_times[i]
            nn_time = checkpoint_times[i]
            reward = np.max([nn_time - baseline_time, 0])
            rewards_history[nn_time] = reward
        
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

            reward = rewards_history[i]

            # Speed
            delta_speed = next_speed - speed

            # if delta_speed == 0 and speed == 0:
            #     # Penalize for not moving
            #     reward -= 1000.0
            # if delta_speed < 0:
            #     if next_speed <= 0:
            #         # Penalize going backwards or not moving
            #         reward -= 100.0
            #     else:
            #         if forward_distance < 5.0:
            #             # reward for slowing down when approaching wall
            #             reward += 2.0
            #         else:
            #             # penalize for slowing down when not approaching wall
            #             reward -= 2.0*(5 - forward_distance)
            # else:
            #     if forward_distance < 5.0:
            #         # penalize for not slowing downe when approaching wall
            #         reward -= 2.0*(5 - forward_distance)
            #     else:
            #         # reward for not slowing down when not approaching wall
            #         reward += 2.0
            
            # Steering
            delta_steering_angle = next_steering_angle - steering_angle
            side_distance_diff = left_distance - right_distance
            # Define tolerance for "centered"
            center_tolerance = 0.1
            
            # left distance > right_distance
            if side_distance_diff > center_tolerance:
                if delta_steering_angle < 0.0:
                    # Reward for steering left when close to a right wall
                    reward += 0.1*(5 - right_distance)
                elif delta_steering_angle > 0.0:
                    # Penalize for steering right when close to right wall
                    # reward -= 0.1*(5 - right_distance)
                    pass
            elif side_distance_diff < -center_tolerance:
                if delta_steering_angle < 0.0:
                    # Penalize for steering left when close to a left wall
                    # reward -= 0.1*(5 - left_distance)
                    pass
                elif delta_steering_angle > 0.0:
                    # Reward for steering right when close to left wall
                    reward += 0.1*(5 - left_distance)
            else:
                # Penalize for steering towards wall in straight sections
                reward -= 0.1 * abs(steering_angle)

            # Small reward for moving
            if forward_distance > 1.0:
                reward += np.max([0.2*speed, 0.2*5.0])

            if speed < 1.0:
                if delta_speed < 0.0:
                    reward -= 10.0
                else:
                    reward += 0.5*delta_speed

            if terminated:
                reward -= 20.0

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
        grads = tape.gradient(loss_value, model.trainable_variables)
        none_count = sum(1 for g in grads if g is None)
        # sim.print(f"DEBUG: total grads = {len(grads)}, None grads = {none_count}")

        for i, (g, v) in enumerate(zip(grads, model.trainable_variables)):
            name = v.name
            if g is None:
                # sim.print(f"[{i}] VAR {name}: grad = None, shape={v.shape}, dtype={v.dtype}")
                pass
            else:
                try:
                    # eager numeric check
                    s = float(tf.reduce_sum(tf.abs(g)).numpy())
                    mn = float(tf.reduce_min(g).numpy())
                    mx = float(tf.reduce_max(g).numpy())
                    # sim.print(f"[{i}] VAR {name}: grad shape={g.shape}, dtype={g.dtype}, sum_abs={s:.6g}, min={mn:.6g}, max={mx:.6g}")
                except Exception as e:
                    # sim.print(f"[{i}] VAR {name}: grad present but numeric read failed: {e}")
                    pass
        
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        sim.print(f"-reward = {episode_reward}, loss = {loss_value}")