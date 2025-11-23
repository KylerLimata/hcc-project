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
        engine_power = 0.0
        breaking_power = 0.0

        if speed_diff < 1.0:
            engine_power = 1.0
        elif speed_diff > 1.0:
            breaking_power = 1.0

        side_distance_diff = left_distance - right_distance
        steering_power = 0.0
        side_distance_diff_normalized = max(-1.0, min(1.0, side_distance_diff / 5.0))
        min_steering_angle = -30.0*(math.pi/180.0)
        max_steering_angle = 30.0*(math.pi/180.0)
        target_steering_angle = min_steering_angle + (side_distance_diff_normalized + 1.0) * ((max_steering_angle - min_steering_angle) / 2.0)
        steering_angle_diff = steering_angle - target_steering_angle

        if steering_angle_diff > 1.0*(math.pi/180.0):
            steering_power = -1.0
        elif steering_angle_diff < -1.0*(math.pi/180.0):
            steering_power = 1.0
            
        return [engine_power, breaking_power, steering_power]
    
class NNAgent:
    def __init__(self, model, num_steering_actions: int, num_engine_actions: int):
        import tensorflow as tf

        self.model = model
        self.num_steering_actions = num_steering_actions
        self.num_engine_actions = num_engine_actions
        self.steering_action_probs_history = []
        self.engine_action_probs_history = []
        self.critic_value_history = []

        # Persistent buffer for input state (shape [1, 5])
        self.full_state = tf.Variable(
            tf.zeros((1, 5), dtype=tf.float32), trainable=False, name="full_state"
        )

        # Precompile model call with fixed input signature
        self._eval_model = tf.function(
            self.model,
            jit_compile=True,
            input_signature=[tf.TensorSpec(shape=(1, 5), dtype=tf.float32)],
        )

    def eval(self, inputs: list[float], state: list[float]):
        import keras
        from keras import ops
        import numpy as np

        full_state_values = np.array(inputs + state, dtype=np.float32).reshape(1, 5)
        self.full_state.assign(full_state_values)

        steering_action_probs, engine_action_probs, critic_value = self._eval_model(self.full_state)
        steering_action = np.random.choice(self.num_steering_actions, p=np.squeeze(steering_action_probs))
        engine_action = np.random.choice(self.num_engine_actions, p=np.squeeze(engine_action_probs))

        self.steering_action_probs_history.append(ops.log(steering_action_probs[0, steering_action]))
        self.engine_action_probs_history.append(ops.log(engine_action_probs[0, engine_action]))
        self.critic_value_history.append(critic_value[0, 0])

        engine_power = engine_action - 1.0
        steering_direction = steering_action - 1.0

        return [steering_direction, engine_power]

class FastNNAgent:
    def __init__(
            self, 
            model, 
            num_steering_actions: int, 
            num_engine_actions: int
            ):
        import tensorflow as tf
        # Upack model weights
        # Weights evaluated with numpy for performance
        layers = model.layers

        self.hidden_layers = []
        for layer in layers:
            if isinstance(layer, tf.keras.layers.Dense) and "out" not in layer.name:
                weights = layer.get_weights()
                if len(weights) == 2:
                    self.hidden_layers.append((weights[0], weights[1], layer.activation))

        self.output_layers = []
        for layer in layers:
            if isinstance(layer, tf.keras.layers.Dense) and "out" in layer.name:
                weights = layer.get_weights()
                if len(weights) == 2:
                    self.output_layers.append((weights[0], weights[1], layer.activation))
        
        self.num_steering_actions = num_steering_actions
        self.num_engine_actions = num_engine_actions
        self.steering_action_probs_history = []
        self.engine_action_probs_history = []
        self.critic_value_history = []

    def apply_activation(self, x, activation):
        import tensorflow as tf
        import numpy as np

        if activation == tf.keras.activations.relu:
            return np.maximum(0, x)
        elif activation == tf.keras.activations.tanh:
            return np.tanh(x)
        elif activation == tf.keras.activations.sigmoid:
            return 1 / (1 + np.exp(-x))
        elif activation == tf.keras.activations.linear:
            return x
        elif activation == tf.keras.activations.softmax:
            # Stable softmax
            z = x - np.max(x, axis=-1, keepdims=True)
            e = np.exp(z)
            return e / np.sum(e, axis=-1, keepdims=True)
        else:
            raise NotImplementedError(f"Unsupported activation: {activation}")
    
    def eval(self, inputs: list[float], state: list[float]):
        import numpy as np

        # Convert inputs and state into single numpy array
        x = np.array(inputs + state, dtype=np.float32).reshape(1, -1)
        
        # Forward pass through hidden layers
        for W, b, activation in self.hidden_layers:
            xprime = np.dot(x, W) + b
            x = self.apply_activation(xprime, activation)
        
        # Forward pass through outputs
        Y = []
        for W, b, activation in self.output_layers:
            yprime = np.dot(x, W) + b
            y = self.apply_activation(yprime, activation)
            Y.append(y[0])

        steering_action_probs = Y[0]
        engine_action_probs = Y[1]
        critic_value = Y[2]

        # Sample actions
        steering_action = np.random.choice(self.num_steering_actions, p=steering_action_probs)
        pedal_action = np.random.choice(self.num_engine_actions, p=engine_action_probs)

        # Log probabilities and update histories
        self.steering_action_probs_history.append(np.log(steering_action_probs[steering_action]))
        self.engine_action_probs_history.append(np.log(engine_action_probs[pedal_action]))
        self.critic_value_history.append(critic_value)

        engine_power = 0.0
        breaking_power = 0.0
        steering_power = steering_action - 1.0

        if pedal_action == 1:
            engine_power = 1.0
        elif pedal_action == 2:
            breaking_power = 1.0

        return [engine_power, breaking_power, steering_power]
    

class NewFastNNAgent:
    def __init__(
            self, 
            model, 
            num_steering_actions: int, 
            num_engine_actions: int
            ):
        import tensorflow as tf
        # Upack model weights
        # Weights evaluated with numpy for performance
        layers = model.layers

        self.hidden_layers = []
        for layer in layers:
            if isinstance(layer, tf.keras.layers.Dense) and "out" not in layer.name:
                weights = layer.get_weights()
                if len(weights) == 2:
                    self.hidden_layers.append((weights[0], weights[1], layer.activation))

        self.output_layers = []
        for layer in layers:
            if isinstance(layer, tf.keras.layers.Dense) and "out" in layer.name:
                weights = layer.get_weights()
                if len(weights) == 2:
                    self.output_layers.append((weights[0], weights[1], layer.activation))
        
        self.num_steering_actions = num_steering_actions
        self.num_engine_actions = num_engine_actions
        self.state_history = []
        self.action_history = []

    def apply_activation(self, x, activation):
        import tensorflow as tf
        import numpy as np

        if activation == tf.keras.activations.relu:
            return np.maximum(0, x)
        elif activation == tf.keras.activations.tanh:
            return np.tanh(x)
        elif activation == tf.keras.activations.sigmoid:
            return 1 / (1 + np.exp(-x))
        elif activation == tf.keras.activations.linear:
            return x
        elif activation == tf.keras.activations.softmax:
            # Stable softmax
            z = x - np.max(x, axis=-1, keepdims=True)
            e = np.exp(z)
            return e / np.sum(e, axis=-1, keepdims=True)
        else:
            raise NotImplementedError(f"Unsupported activation: {activation}")
    
    def eval(self, inputs: list[float], state: list[float]):
        import numpy as np

        # Convert inputs and state into single numpy array
        x = np.array(inputs + state, dtype=np.float32).reshape(1, -1)
        
        # Forward pass through hidden layers
        for W, b, activation in self.hidden_layers:
            xprime = np.dot(x, W) + b
            x = self.apply_activation(xprime, activation)
        
        # Forward pass through outputs
        Y = []
        for W, b, activation in self.output_layers:
            yprime = np.dot(x, W) + b
            y = self.apply_activation(yprime, activation)
            Y.append(y[0])

        steering_action_probs = Y[0]
        engine_action_probs = Y[1]
        critic_value = Y[2]

        # Sample actions
        steering_action = np.random.choice(self.num_steering_actions, p=steering_action_probs)
        pedal_action = np.random.choice(self.num_engine_actions, p=engine_action_probs)

        # Update histories
        self.state_history.append(inputs + state)
        self.action_history.append((steering_action, pedal_action))

        engine_power = 0.0
        breaking_power = 0.0
        steering_power = steering_action - 1.0

        if pedal_action == 1:
            engine_power = 1.0
        elif pedal_action == 2:
            breaking_power = 1.0

        return [engine_power, breaking_power, steering_power]
    
