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
critic_steering = layers.Dense(1, activation="linear", name="critic_steering_out")(common)
critic_engine = layers.Dense(1, activation="linear", name="critic_engine_out")(common)

model = keras.Model(inputs=inputs, outputs=[action_steering, action_engine, critic_steering, critic_engine])

# Train neural network agent
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
steering_action_probs_history = []
engine_action_probs_history = []
steering_critic_value_history = []
engine_critic_value_history = []
rewards_history = []
steering_rewards_history = []
engine_rewards_history = []
episode_count = 0

while episode_count < max_episodes:
    sim.load_environment("training_environment_new")

    episode_engine_reward = 0.0
    episode_steering_reward = 0.0
    # Create agent and run episode to get states
    agent = agents.NNAgent(model, num_steering_actions, num_engine_actions)
    sim.print(f"Running training episode {episode_count + 1}.")
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
        action_steering, action_engine, steering_critic_values, engine_critic_values = model(states_tf, training=True)

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
            base_reward = 0.0

            # --- Base checkpoint reward ---
            if j < len(checkpoint_times) and step > checkpoint_times[j]:
                j += 1

            if j < len(checkpoint_times):
                baseline_time = baseline_checkpoint_times[j]
                nn_time = checkpoint_times[j]
                base_reward = np.maximum(baseline_time - nn_time, 0)

            side_distance_diff = left_distance - right_distance
                
            # Engine Reward
            engine_reward = base_reward

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
            steering_reward = base_reward

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

            # # Append reward
            # if step % 10 == 0:
            #     sim.print(f"state = ({speed} m/s, {steering_angle} rad), input = ({left_distance} m, {forward_distance} m, {right_distance} m)")
            #     sim.print(f"action = ({engine_power}, {steering_power}), reward = ({engine_reward}, {steering_reward})")
            steering_rewards_history.append(steering_reward)
            engine_rewards_history.append(engine_reward)

        if terminated:
            steering_rewards_history[-1] -= 100
            engine_rewards_history[-1] -= 100

        episode_engine_reward = sum([r[0] for r in rewards_history])
        episode_steering_reward = sum([r[1] for r in rewards_history])


        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns_steering = []
        returns_engine = []
        discounted_sum_steering = 0
        discounted_sum_engine = 0

        for rs, re in zip(steering_rewards_history[::-1], engine_rewards_history[::-1]):
            discounted_sum_steering = rs + gamma*discounted_sum_steering
            discounted_sum_engine = re + gamma*discounted_sum_engine

            returns_steering.insert(0, discounted_sum_steering)
            returns_engine.insert(0, discounted_sum_engine)

        # Normalize
        returns_steering = np.array(returns_steering)
        returns_steering = (returns_steering - np.mean(returns_steering)) / (np.std(returns_steering) + eps)
        returns_steering = returns_steering.tolist()
        returns_engine = np.array(returns_engine)
        returns_engine = (returns_engine - np.mean(returns_engine)) / (np.std(returns_engine) + eps)
        returns_engine = returns_engine.tolist()

        # Calculating loss values to update the network
        history = zip(
            steering_action_probs_history,
            engine_action_probs_history,
            steering_critic_values,
            engine_critic_values,
            returns_steering,
            returns_engine
        )

        actor_losses = []
        critic_losses = []

        for logprob_s, logprob_e, val_s, val_e, ret_s, ret_e in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up receiving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff_s = ret_s - val_s
            diff_e = ret_e - val_e
            actor_loss_steering = -logprob_s*diff_s
            actor_loss_engine = -logprob_e*diff_e

            actor_losses.append(actor_loss_steering + actor_loss_engine)
            # The critics must be updated so that they predict a better estimate of
            # the future rewards.
            critic_loss_steering = huber_loss(tf.expand_dims(val_s, 0), tf.expand_dims(ret_s, 0))
            critic_loss_engine = huber_loss(tf.expand_dims(val_e, 0), tf.expand_dims(ret_e, 0))

            critic_losses.append(critic_loss_steering + critic_loss_engine)

        # Backpropagation
        loss_value = tf.add_n(actor_losses) + tf.add_n(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_count += 1

        sim.print(f"rewards = ({episode_engine_reward, episode_steering_reward})")