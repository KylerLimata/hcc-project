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
    
# Evaluate the baseline agent
sim.load_environment("training_environment")
handle = sim.run_episode(agents.BaselineAgent(), max_seconds_per_episode*60)
sim.print("Running episode for baseline agent.")


while True:
    if handle.is_done():
        break

baseline_checkpoint_times, terminated = handle.get_result()
sim.print("Episode for baseline agent complete.")

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

sim.print("Here!")

while episode_count < max_episodes:
    sim.load_environment("training_environment")
    episode_reward = 0
    
    sim.print(f"Running training episode {episode_count + 1}.")

    with tf.GradientTape() as tape:
        # Create agent and run episode
        agent = agents.NNAgent(model, num_engine_actions, num_engine_actions)
        handle = sim.run_episode(agent, max_seconds_per_episode*60)

        sim.print(" Episode started.")

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
            reward = nn_time - baseline_time

            # Reward every step until it hits the checkpoint
            while len(rewards_history) < nn_time:
                rewards_history.append(reward)

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
    
        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(steering_action_probs_history, engine_action_probs_history, critic_value_history, returns)
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
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        steering_action_probs_history.clear()
        engine_action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    sim.print(f" reward: {episode_reward}")