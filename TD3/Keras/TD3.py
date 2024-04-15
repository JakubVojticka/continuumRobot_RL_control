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
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
            

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1][0]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):

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

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model1([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
 
def get_actor():
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003) 
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs) 
    out = layers.Dense(256, activation="relu")(out) 
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    outputs = outputs * upper_bound 
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic1():

    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)

    outputs = layers.Dense(1)(out) 
    model1 = tf.keras.Model([state_input, action_input], outputs)

    return model1

def get_critic2():
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)

    outputs = layers.Dense(1)(out) 

    model2 = tf.keras.Model([state_input, action_input], outputs)

    return model2

def policy(state, noise_object, add_noise=True):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object() 

    if add_noise:
        sampled_actions = sampled_actions.numpy() + noise
    
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

actor_model = get_actor()
critic_model1 = get_critic1()
critic_model2 = get_critic2()

target_actor = get_actor()
target_critic1 = get_critic1()
target_critic2 = get_critic2()

target_actor.set_weights(actor_model.get_weights())
target_critic1.set_weights(critic_model1.get_weights())
target_critic2.set_weights(critic_model2.get_weights())

gamma = 0.99            
tau = 5e-3              

critic_lr = 1e-2        # learning rate of the critic
actor_lr = 1e-3         # learning rate of the actor

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 250

buffer = Buffer(int(5e5), 64)

ep_reward_list_1 = []
avg_reward_list_1 = []
counter = 0
avg_reward = 0

TRAIN = False  # Setting to False for evaluation

if TRAIN:
    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions))
    
    for ep in range(total_episodes):
        
        prev_state = env.reset()  # starting position is random (within task space)
        
        if ep % 100 == 0:
            print('Episode Number', ep)
            print("Initial Position is", prev_state[0:2])
            print("===============================================================")
            print("Target Position is", prev_state[2:4])
            print("===============================================================")
            print("Initial Kappas are ", [env.kappa1, env.kappa2, env.kappa3])
            print("===============================================================")
            print("Goal Kappas are ", [env.target_k1, env.target_k2, env.target_k3])
            print("===============================================================")
        
        time.sleep(2)  
        episodic_reward = 0
    
        for i in range(1000):
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = policy(tf_prev_state, ou_noise)
    
            state, reward, done, info = env.step_minus_euclidean_square(action[0])  # -e^2
            
            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward
    
            buffer.learn()
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic1.variables, critic_model1.variables, tau)
            update_target(target_critic2.variables, critic_model2.variables, tau)    
            
            if done:
                counter += 1
                break
    
            prev_state = state
            
            print("Episode Number {0} and {1}th action".format(ep,i))
            print("Goal Position",prev_state[2:4])
            # # print("Previous Error: {0}, Error: {1}, Current State: {2}".format(env.previous_error, env.error, prev_state)) # for step_1
            #print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), prev_state)) # for step_2

            if reward < 0:
                print("Error: N/A, Current State:", prev_state)
            else:
                print("Error: {0}, Current State: {1}".format(math.sqrt(reward), prev_state))       

            print("Action: {0},  Kappas {1}".format(action, [env.kappa1,env.kappa2,env.kappa3]))
            print("Reward is ", reward)
            print("{0} times robot reached to the target".format(counter))
            print("Avg Reward is {0}, Episodic Reward is {1}".format(avg_reward,episodic_reward))
            print("--------------------------------------------------------------------------------")
    
        ep_reward_list_1.append(episodic_reward)
    
        avg_reward = np.mean(ep_reward_list_1[-100:])
        if ep % 100 == 0:
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            time.sleep(0.5)
        avg_reward_list_1.append(avg_reward)
    
    print(f'{counter} times robot reached the target point in total {total_episodes} episodes')
    
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(avg_reward_list_1)+1), avg_reward_list_1)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")

    with open('avg_reward_list.pickle', 'wb') as f:
        pickle.dump(avg_reward_list_1, f, pickle.HIGHEST_PROTOCOL)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(ep_reward_list_1)+1), ep_reward_list_1)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.show()

    with open('ep_reward_list.pickle', 'wb') as f:
        pickle.dump(ep_reward_list_1, f, pickle.HIGHEST_PROTOCOL)

    actor_model.save_weights("td3_actor.h5")
    critic_model1.save_weights("td3_critic1.h5")
    critic_model2.save_weights("td3_critic2.h5")
    target_actor.save_weights("td3_target_actor.h5")
    target_critic1.save_weights("td3_target_critic1.h5")
    target_critic2.save_weights("td3_target_critic2.h5")
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
