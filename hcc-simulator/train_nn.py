import numpy as np
import keras
from keras import ops
from keras import layers
import tensorflow as tf
import agents

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_seconds_per_episode = 2
max_episodes = 10
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
    
# Evaluate the baseline agent
sim.load_environment("training_environment")
handle = sim.run_episode(agents.BaselineAgent(), max_seconds_per_episode*60)
sim.print("Running episode for baseline agent.")


while True:
    if handle.is_done():
        break

baseline_checkpoint_times, _, _ = handle.get_result()
sim.print("Episode for baseline agent complete.")

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

while episode_count < max_episodes:
    episode_reward = 0
    # Create agent and run episode to get states
    agent = agents.NewFastNNAgent(model, num_steering_actions, num_engine_actions)
    sim.print(f"Running training episode {episode_count + 1}.")
    handle = sim.run_episode(agent, max_seconds_per_episode*60)

    sim.print(" Running episode simulation.")

    while True:
        if handle.is_done():
            break

    sim.print(" Finished simulation")

    # Upack results
    checkpoint_times, terminated, end_step = handle.get_result()
    state_history = agent.state_history
    action_history = agent.action_history
    states_tf = tf.convert_to_tensor(agent.state_history, dtype=tf.float32)
    rewards_history = [0]*end_step

    sim.print(f"len(checkpoint_times): {len(checkpoint_times)}")

    with tf.GradientTape() as tape:
        # Use actual model to calculate states, needed to compute gradients
        action_steering, action_engine, critic_values = model(states_tf, training=True)

        # Extract chosen action probabilities
        steering_indices = tf.constant([a[0] for a in agent.action_history], dtype=tf.int32)
        engine_indices   = tf.constant([a[1] for a in agent.action_history], dtype=tf.int32)

        steering_action_probs_history = tf.gather(action_steering, steering_indices, axis=1, batch_dims=1)
        engine_action_probs_history   = tf.gather(action_engine, engine_indices, axis=1, batch_dims=1)

        # Fill rewards history
        for i in range(len(checkpoint_times)):
            baseline_time = baseline_checkpoint_times[i]
            nn_time = checkpoint_times[i]
            reward = nn_time - baseline_time
            rewards_history[nn_time] = reward
            episode_reward += reward

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

        sim.print(" normalizing returns...")
    
        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        sim.print(" computing loss...")

        # Calculating loss values to update our network
        history = zip(steering_action_probs_history, engine_action_probs_history, critic_values, returns)
        actor_losses = []
        critic_losses = []

        sim.print(f"len(steering_action_probs_history): {len(steering_action_probs_history)}")
        sim.print(f"len(engine_action_probs_history): {len(engine_action_probs_history)}")
        sim.print(f"critic_values shape: {getattr(critic_values, 'shape', None)}")
        sim.print(f"len(returns): {len(returns)}")


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
        sim.print(f"actor_losses length: {len(actor_losses)}")
        sim.print(f"critic_losses length: {len(critic_losses)}")

        for i, (a, c) in enumerate(zip(actor_losses, critic_losses)):
            sim.print(f"[{i}] actor_loss type={type(a)}, shape={getattr(a, 'shape', None)}")
            sim.print(f"[{i}] critic_loss type={type(c)}, shape={getattr(c, 'shape', None)}")    
        
        sim.print(" Performing backpropagation...")
        loss_value = tf.add_n(actor_losses) + tf.add_n(critic_losses)
        sim.print(f"loss_value dtype/shape: {type(loss_value)}, {getattr(loss_value,'shape',None)}")
        try:
            tf.debugging.check_numerics(loss_value, "Loss has NaN/Inf")
        except Exception as e:
            sim.print(f"Loss numerics check failed: {e}")
        sim.print(" Computing gradients...")
        try:
            grads = tape.gradient(loss_value, model.trainable_variables)
        except Exception as e:
            sim.print(f"gradient computation failed: {e}")
            raise
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