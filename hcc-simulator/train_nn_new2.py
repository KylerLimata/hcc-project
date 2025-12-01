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
max_episodes = 5
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
    
# Load baseline checkpoint times
baseline_checkpoint_times = np.load('baseline_checkpoint_times.npy')

# Setup actor critic network
num_inputs = 5
num_steering_actions = 3
num_engine_actions = 3
num_hidden = 256

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
    sim.load_environment("training_environment_new")

    episode_reward = 0
    # Create agent and run episode to get states
    agent = agents.NNAgent(model, num_steering_actions, num_engine_actions)
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
    rewards_history = []

    sim.print(f" Completed in {end_step} steps")

    with tf.GradientTape() as tape:
        # Use actual model to calculate states, needed to compute gradients
        action_steering, action_engine, critic_values = model(states_tf, training=True)

        # Extract chosen action probabilities
        steering_indices = tf.constant([a[0] for a in agent.action_history], dtype=tf.int32)
        engine_indices   = tf.constant([a[1] for a in agent.action_history], dtype=tf.int32)

        steering_action_probs_history = tf.gather(action_steering, steering_indices, axis=1, batch_dims=1)
        engine_action_probs_history   = tf.gather(action_engine, engine_indices, axis=1, batch_dims=1)

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
            steering_action = action[0]
            engine_action = action[1]

            # Engine and steering power
            engine_power = 0.0
            steering_power = 0.0

            if engine_action == 0:
                engine_power = -1.0
            elif engine_action == 1:
                engine_power = 0.0
            else:
                engine_power = 1.0

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
                reward = np.maximum(baseline_time - nn_time, 0)

            side_distance_diff = left_distance - right_distance
                
            # Engine Reward
            engine_reward = 0.0    

            if engine_power == 1.0:
                if abs(side_distance_diff) > center_tolerance:
                    if left_distance < right_distance and steering_power == -1.0:
                        engine_reward -= 1.0*(5.0 - min(left_distance, right_distance))
                    elif right_distance < left_distance and steering_action == 1.0:
                        engine_reward -= 1.0*(5.0 - min(left_distance, right_distance))
                else:
                    engine_reward += 1.0*forward_distance - 1.0
            if engine_power == -1.0:
                if speed > 1.0:
                    engine_reward += 1.0*(5-forward_distance)
                else:
                    engine_reward -= 2.0
            if engine_power == 0.0:
                if speed < 0.0:
                    engine_reward -= 2.0

            # Steering rewards/penalties
            center_tolerance = 0.1
            steering_reward = 0.0

            # Turning
            if abs(side_distance_diff) > center_tolerance:
                # Turning left
                if left_distance < right_distance:
                    steering_reward += (2 if steering_power < 0 else -2)
                # Turning right
                if right_distance < left_distance:
                    steering_reward += (2 if steering_power > 0 else -2)
            # Not turning
            else:
                if steering_power == 0.0:
                    steering_reward += 2

            reward += steering_reward + engine_reward
                    
            # Append reward
            if step % 10 == 0:
                sim.print(f"state = ({speed} m/s, {steering_angle} rad), input = ({left_distance} m, {forward_distance} m, {right_distance} m)")
                sim.print(f"action = ({engine_power}, {steering_power}), reward = ({engine_reward}, {steering_reward})")
            rewards_history.append(reward)

        if terminated:
            rewards_history[-1] -= 100

        episode_reward = sum(rewards_history)

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

        sim.print(f" reward = {episode_reward}, loss = {loss_value}")