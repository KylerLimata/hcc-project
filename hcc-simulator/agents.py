class DebugAgent:
    def __init__(self):
        self.states = []

    def eval(self, inputs: list[float], state: list[float]):
        self.states.append(state)

        return [1.0, -1.0]

class BaselineAgent:
    def __init__(self):
        pass
    
    def eval(self, inputs: list[float], state: list[float]):
        import math

        # Unpack input vec
        left_dist = inputs[0]
        left_forward_dist = inputs[1]
        forward_dist = inputs[2]
        right_forward_dist = inputs[3]
        right_dist = inputs[4]
        # Unpack state vec
        speed = state[0]
        steering_angle = state[1]

        forward_side_diff = left_forward_dist - right_forward_dist
        forward_side_sum = left_forward_dist + right_forward_dist
        forward_side_dist = forward_dist

        if left_forward_dist < right_forward_dist:
            forward_side_dist = left_forward_dist
        else:
            right_forward_dist = right_forward_dist

        projected_dist = forward_side_dist*math.cos(math.pi/6.0)
        w = min(max(forward_side_diff/forward_side_sum, 0.0), 1.0)
        forward_dist_interp = (1 - w)*forward_dist + w*projected_dist

        target_speed = 5*(forward_dist_interp)
        speed_err = speed - target_speed
        engine_power = 0.0

        if speed_err < 1.0:
            engine_power = 1.0
        elif speed_err > 1.0:
            engine_power = -1.0

        side_dist_diff_norm = max(-1.0, min(1.0, (left_dist - right_dist)/10.0))
        forward_dist_diff_norm = max(-1.0, min(1.0, (forward_side_diff)/10.0))
        min_steering = -30.0*(math.pi/180.0)
        max_steering = 30.0*(math.pi/180.0)
        alpha = min_steering + (side_dist_diff_norm + 1.0) * ((max_steering - min_steering) / 2.0)
        beta = min_steering + (forward_dist_diff_norm + 1.0) * ((max_steering - min_steering) / 2.0)
        target_steering_angle = 0.75*alpha + 0.25*beta
        steering_err = steering_angle - target_steering_angle

        steering_power = 0.0

        if steering_err > 1.0*(math.pi/180.0):
            steering_power = -1.0
        elif steering_err < -1.0*(math.pi/180.0):
            steering_power = 1.0
            
        return [engine_power, steering_power]
    
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

class NNAgent:
    def __init__(
            self, 
            model, 
            num_steering_actions: int, 
            num_engine_actions: int
            ):
        import tensorflow as tf
        import numpy as np
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
        self.rng = np.random.default_rng(8282)

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
        steering_action = self.rng.choice(self.num_steering_actions, p=steering_action_probs)
        engine_action = self.rng.choice(self.num_engine_actions, p=engine_action_probs)

        # Update histories
        self.state_history.append(inputs + state)
        self.action_history.append((steering_action, engine_action))

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

        return [engine_power, steering_power]
    
class NNEngineAgent:
    def __init__(
            self, 
            model, 
            num_steering_actions: int, 
            num_engine_actions: int
            ):
        import tensorflow as tf
        import numpy as np
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
        self.rng = np.random.default_rng(8282)

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
        import math

        # Unpack input vec
        left_distance = inputs[0]
        right_distance = inputs[2]
        # Unpack state vec
        steering_angle = state[1]

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

        engine_action_probs = Y[0]

        # Sample actions
        engine_action = self.rng.choice(self.num_engine_actions, p=engine_action_probs)

        # Update histories
        self.state_history.append(inputs + state)
        self.action_history.append(engine_action)

        engine_power = 0.0
        steering_power = 0.0

        if engine_action == 0:
            engine_power = -1.0
        elif engine_action == 1:
            engine_power = 0.0
        else:
            engine_power = 1.0

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

        return [engine_power, steering_power]

class NNSteeringAgent:
    def __init__(
            self, 
            model, 
            num_steering_actions: int
            ):
        import tensorflow as tf
        import numpy as np
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
        self.state_history = []
        self.action_history = []
        self.rng = np.random.default_rng(8282)

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

        # Sample actions
        steering_action = self.rng.choice(self.num_steering_actions, p=steering_action_probs)

        # Update histories
        self.state_history.append(inputs + state)
        self.action_history.append(steering_action)

        steering_power = 0.0

        if steering_action == 0:
            steering_power = -1.0
        elif steering_action == 1:
            steering_power = 0.0
        else:
            steering_power = 1.0


        # Unpack input vec
        forward_distance = inputs[2]
        # Unpack state vec
        speed = state[0]

        target_speed = 20*(forward_distance)
        speed_diff = speed - target_speed
        engine_power = 0.0

        if speed_diff < 1.0:
            engine_power = 1.0
        elif speed_diff > 1.0:
            engine_power = -1.0

        return [engine_power, steering_power]
    
