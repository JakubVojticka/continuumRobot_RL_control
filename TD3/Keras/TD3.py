import sys
sys.path.append('../')
sys.path.append('../RL')
sys.path.append('../Tests')

import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from environment import continuumEnv
from tensorflow.keras import layers

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

env = continuumEnv()

num_states = env.observation_space.shape[0] * 2 
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))
start_time = time.time()

class OUActionNoise:

    #It creates a noise process that is correlated with the previous noise value 

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

# Define replay buffer with policy delay
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64, policy_delay=2):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.buffer_counter = 0
        self.update_counter = 0  # Track critic updates

        # Initialize buffers for states, actions, rewards, and next states
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Record data to buffer
    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1][0]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    # Update models based on sampled data
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, update_actor=False):
        # Critic update
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic1(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model1([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model1.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model1.trainable_variables)
        )

        # Actor update, if policy delay condition is met
        if update_actor:
            with tf.GradientTape() as tape:
                actions = actor_model(state_batch, training=True)
                critic_value = critic_model1([state_batch, actions], training=True)
                actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
            actor_optimizer.apply_gradients(
                zip(actor_grad, actor_model.trainable_variables)
            )

            # Update target networks
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic1.variables, critic_model1.variables, tau)
            update_target(target_critic2.variables, critic_model2.variables, tau)

    # Perform learning by sampling from buffer
    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.cast(tf.convert_to_tensor(self.reward_buffer[batch_indices]), dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Determine if we should update the actor
        update_actor = (self.update_counter % self.policy_delay) == 0

        # Call the update function
        self.update(state_batch, action_batch, reward_batch, next_state_batch, update_actor)

        self.update_counter += 1  # Increment the update counter
@tf.function
def update_target(target_weights, weights, tau):
    # This function updates the target network's weights using a soft update formula.
    # It blends the target network's existing weights with the updated weights using a parameter tau.
    # The tau parameter defines the degree of mixing between the new weights and the existing weights.
    # A higher tau results in a more significant update, while a lower tau keeps the weights closer to their existing state.

    for (a, b) in zip(target_weights, weights):
        # The new weight for 'a' is calculated by taking a portion 'tau' of 'b' (the new weights),
        # and combining it with a portion '1 - tau' of 'a' (the existing weights).
        # This creates a smooth transition, preventing abrupt changes in the target network.
        a.assign(b * tau + a * (1 - tau))
 
# This function creates the actor model for a reinforcement learning agent.
# It outputs actions in the range of [-upper_bound, upper_bound] with a tanh activation.
def get_actor():
    # Initialize the final layer with a narrow range to avoid large initial outputs.
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    # Input layer for the state variables.
    inputs = layers.Input(shape=(num_states,))
    
    # Hidden layers with ReLU activation to learn feature representations.
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    
    # Output layer with tanh activation to ensure actions are between -1 and 1.
    # The kernel initializer keeps the weights small initially to avoid instability.
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    # Scale the output to the desired action range.
    outputs = outputs * upper_bound
    
    # Create and return the model.
    model = tf.keras.Model(inputs, outputs)
    return model


# This function creates the first critic model.
# Critic models are used to evaluate the value of state-action pairs.
def get_critic1():
    # Input layer for state variables.
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Input layer for action variables.
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Concatenate state and action representations.
    concat = layers.Concatenate()([state_out, action_out])

    # Further processing layers to compute the value.
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    
    # Output layer to predict the value of the state-action pair.
    outputs = layers.Dense(1)(out)
    
    # Create and return the critic model.
    model1 = tf.keras.Model([state_input, action_input], outputs)
    return model1


# This function creates the second critic model.
# Having two critic models helps with stability during training by reducing bias.
def get_critic2():
    # Similar architecture to the first critic model.
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Concatenate state and action representations.
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    
    outputs = layers.Dense(1)(out)
    
    model2 = tf.keras.Model([state_input, action_input], outputs)
    return model2


# This function generates a policy based on the actor model and adds noise for exploration.
def policy(state, noise_object, add_noise=True):
    # Generate actions from the actor model.
    sampled_actions = tf.squeeze(actor_model(state))
    
    # Get noise from the noise object.
    noise = noise_object()
    
    # If add_noise is true, add the noise to the actions for exploration.
    if add_noise:
        sampled_actions = sampled_actions.numpy() + noise
    
    # Ensure the actions are within the legal bounds.
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    
    # Return the valid action.
    return [np.squeeze(legal_action)]

# Create the actor and critic models, as well as their corresponding target models.
actor_model = get_actor()  # Main actor model for policy generation
critic_model1 = get_critic1()  # First critic model to evaluate state-action pairs
critic_model2 = get_critic2()  # Second critic model for stability and reduced bias

target_actor = get_actor()  # Target actor model for soft updates
target_critic1 = get_critic1()  # Target critic model (1) for soft updates
target_critic2 = get_critic2()  # Target critic model (2) for soft updates

# Initialize the target models' weights to match the original models.
# This ensures they start with the same parameters.
target_actor.set_weights(actor_model.get_weights())
target_critic1.set_weights(critic_model1.get_weights())
target_critic2.set_weights(critic_model2.get_weights())

# Define hyperparameters for the reinforcement learning setup.
gamma = 0.99  # Discount factor for future rewards, typically close to 1
tau = 5e-3  # Soft update rate for target models, a small value for smooth updates

# Learning rates for the optimizer.
critic_lr = 1e-2  # Learning rate for the critic optimizer, can be higher than actor's
actor_lr = 1e-3  # Learning rate for the actor optimizer, generally lower for stability

# Initialize optimizers for the actor and critic models.
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)  # Optimizer for critics
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)  # Optimizer for actor

# Total number of episodes to run the reinforcement learning agent.
total_episodes = 100  # Total episodes for training, can vary based on task complexity

# Create a buffer to store experiences for the replay memory.
buffer = Buffer(int(5e5), 64)  # Replay buffer with a capacity of 500,000 and batch size of 64

# Lists to track the reward and average reward per episode.
ep_reward_list = []  # List to store total rewards per episode
avg_reward_list = []  # List to store average rewards for moving average
counter = 0  # Counter to track number of episodes processed
avg_reward = 0  # Variable to store the running average of rewards

TRAIN = False # Setting to False for evaluation

if TRAIN:
    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions))
    
    for ep in range(total_episodes):
        
        prev_state = env.reset() # starting postion is random (within task space)
        if ep % 100 == 0:
            print('Episode Number',ep)
            print("Initial Position is",prev_state[0:2])
            print("===============================================================")
            print("Target Position is",prev_state[2:4])
            print("===============================================================")
            print("Initial Kappas are ",[env.kappa1,env.kappa2,env.kappa3])
            print("===============================================================")
            print("Goal Kappas are ",[env.target_k1,env.target_k2,env.target_k3])
            print("===============================================================")
        
        time.sleep(2) 
        episodic_reward = 0
    
        # while True:
        for i in range(200):

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = policy(tf_prev_state, ou_noise)
    
            # Recieve state and reward from environment.
            state, reward, done, info = env.step_minus_euclidean_square(action[0]) # -e^2 
            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward
    
            buffer.learn()
            if done:
                counter += 1
                break
    
            prev_state = state
            print("Episode Number {0} and {1}th action".format(ep,i))
            print("Goal Position",prev_state[2:4])
            print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), prev_state)) # for step_2
            print("Action: {0},  Kappas {1}".format(action, [env.kappa1,env.kappa2,env.kappa3]))
            print("Reward is ", reward)
            print("{0} times robot reached to the target".format(counter))
            print("Avg Reward is {0}, Episodic Reward is {1}".format(avg_reward,episodic_reward))
            print("--------------------------------------------------------------------------------")
    
        ep_reward_list.append(episodic_reward)
    
        # Mean of 250 episodes
        avg_reward = np.mean(ep_reward_list[-100:])
        if ep % 100 == 0:
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            time.sleep(0.5)
        avg_reward_list.append(avg_reward)
    
    print(f'{counter} times robot reached the target point in total {total_episodes} episodes')
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(avg_reward_list)+1), avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")

    with open('avg_reward_list.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(avg_reward_list, f, pickle.HIGHEST_PROTOCOL)
    
    # Episodes versus Rewards
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(ep_reward_list)+1), ep_reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.show()

    with open('ep_reward_list.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(ep_reward_list, f, pickle.HIGHEST_PROTOCOL)

    # Save Weights
    actor_model.save_weights("../Keras/td3_actor.h5")
    critic_model1.save_weights("../Keras/td3_critic1.h5")
    critic_model2.save_weights("../Keras/td3_critic2.h5")
    target_actor.save_weights("../Keras/td3_target_actor.h5")
    target_critic1.save_weights("../Keras/td3_target_critic1.h5")
    target_critic2.save_weights("../Keras/td3_target_critic2.h5")
    end_time = time.time() - start_time
    print('Total Overshoot 0: ', env.overshoot0)
    print('Total Overshoot 1: ', env.overshoot1)
    print('Total Elapsed Time is:', int(end_time)/60)
else:
    actor_model.load_weights("../Keras/td3_actor.h5")
    critic_model1.load_weights("../Keras/td3_critic1.h5")
    critic_model2.load_weights("../Keras/td3_critic2.h5")

    target_actor.load_weights("../Keras/td3_target_actor.h5")
    target_critic1.load_weights("../Keras/td3_target_critic1.h5")
    target_critic2.load_weights("../Keras/td3_target_critic2.h5")