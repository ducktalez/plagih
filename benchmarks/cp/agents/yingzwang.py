import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from collections import deque
import random
import time
from keras import layers, initializers, regularizers
from functools import partial
from pathlib import Path
import csv
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size=int(1e5), random_seed=1234):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            The right side of the deque contains the most recent experiences.
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        random.seed(random_seed)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)

    def add(self, s, a, r, done, s2):
        """Add a new experience to buffer.
        Params
        ======
        s: one state sample, numpy array shape (s_dim,)
        a: one action sample, scalar (for DQN)
        r: one reward sample, scalar
        done: True/False, scalar
        s2: one state sample, numpy array shape (s_dim,)
        """
        e = (s, a, r, done, s2)
        self.buffer.append(e)

    def sample_batch(self, batch_size):
        """Randomly sample a batch of experiences from buffer."""

        # ensure the buffer is large enough for sampleling
        assert (len(self.buffer) >= batch_size)

        # sample a batch
        batch = random.sample(self.buffer, batch_size)

        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states, actions, rewards, dones, next_states = zip(*batch)
        states = np.asarray(states).reshape(batch_size, -1)  # shape (batch_size, s_dim)
        next_states = np.asarray(next_states).reshape(batch_size, -1)  # shape (batch_size, s_dim)
        actions = np.asarray(actions)  # shape (batch_size,), for DQN, action is an int
        rewards = np.asarray(rewards)  # shape (batch_size,)
        dones = np.asarray(dones, dtype=np.uint8)  # shape (batch_size,)
        return states, actions, rewards, dones, next_states


def build_summaries():
    """
    tensorboard summary for monitoring training process
    """

    # performance per episode
    ph_reward = tensorflow.compat.v1.placeholder(tensorflow.float32)
    tensorflow.compat.v1.summary.scalar("Reward_ep", ph_reward)
    ph_Qmax = tensorflow.compat.v1.placeholder(tensorflow.float32)
    tensorflow.compat.v1.summary.scalar("Qmax_ep", ph_Qmax)

    # merge all summary op_dict (must be done at the last step)
    summary_op = tensorflow.compat.v1.summary.merge_all()

    return summary_op, ph_reward, ph_Qmax


def build_net(model_name, state, a_dim, args, trainable):
    """
    neural network model
    model input: state
    model output: Qhat
    """
    h1 = int(args['h1'])
    h2 = int(args['h2'])

    my_dense = partial(layers.Dense, trainable=trainable)
    with tensorflow.compat.v1.variable_scope(model_name):
        net = my_dense(h1, name="l1-dense-{}".format(h1))(state)
        net = layers.Activation('relu', name="relu1")(net)
        net = my_dense(h2, name="MSE-dense-{}".format(h2))(net)
        net = layers.Activation('relu', name="relu2")(net)
        net = my_dense(a_dim, name="l3-dense-{}".format(a_dim))(net)
    Qhat = layers.Activation('linear', name="Qhat")(net)
    nn_params = tensorflow.compat.v1.trainable_variables(scope=model_name)
    return Qhat, nn_params


class DeepQNetwork:
    def __init__(self, sess, a_dim, s_dim, args):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.h1 = args["h1"]
        self.h2 = args["h2"]
        self.lr = args["learning_rate"]
        self.gamma = args["gamma"]
        self.epsilon_start = args["epsilon_start"]
        self.epsilon_stop = args["epsilon_stop"]
        self.epsilon_decay = args["epsilon_decay"]
        self.epsilon = self.epsilon_start  # current exploration probability
        self.update_target_C = args["update_target_C"]
        self.update_target_tau = args['update_target_tau']
        self.learn_step_counter = 0

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
        self.minibatch_size = int(args['minibatch_size'])

        self.s = tensorflow.compat.v1.placeholder(tensorflow.float32, [None, self.s_dim], name='state')  # input State
        self.s_ = tensorflow.compat.v1.placeholder(tensorflow.float32, [None, self.s_dim], name='state_next')  # input Next State
        self.r = tensorflow.compat.v1.placeholder(tensorflow.float32, [None, ], name='reward')  # input Reward
        self.a = tensorflow.compat.v1.placeholder(tensorflow.int32, [None, ], name='action')  # input Action
        self.done = tensorflow.compat.v1.placeholder(tensorflow.float32, [None, ], name='done')

        # initialize NN, self.q shape (batch_size, a_dim)
        self.q, self.nn_params = build_net("DQN", self.s, a_dim, args, trainable=True)
        self.q_, self.nn_params_ = build_net("target_DQN", self.s_, a_dim, args, trainable=False)
        for var in self.nn_params:
            vname = var.name.replace("kernel:0", "W").replace("bias:0", "b")
            tensorflow.compat.v1.summary.histogram(vname, var)

        with tensorflow.compat.v1.variable_scope("Qmax"):
            self.Qmax = tensorflow.reduce_max(self.q_, axis=1)  # shape (batch_size,)

        with tensorflow.compat.v1.variable_scope("yi"):
            self.yi = self.r + self.gamma * self.Qmax * (1 - self.done)  # shape (batch_size,)

        with tensorflow.compat.v1.variable_scope("Qa_all"):
            Qa = tensorflow.Variable(tensorflow.zeros([self.minibatch_size, self.a_dim]))
            for aval in np.arange(self.a_dim):
                tensorflow.compat.v1.summary.histogram("Qa{}".format(aval), Qa[:, aval])
            self.Qa_op = Qa.assign(self.q)

        with tensorflow.compat.v1.variable_scope("Q_at_a"):
            # select the Q value corresponding to the action
            one_hot_actions = tensorflow.one_hot(self.a, self.a_dim)  # shape (batch_size, a_dim)
            q_all = tensorflow.multiply(self.q, one_hot_actions)  # shape (batch_size, a_dim)
            self.q_at_a = tensorflow.reduce_sum(q_all, axis=1)  # shape (batch_size,)

        with tensorflow.compat.v1.variable_scope("loss_MSE"):
            self.loss = tensorflow.compat.v1.losses.mean_squared_error(labels=self.yi, predictions=self.q_at_a)

        with tensorflow.compat.v1.variable_scope("train_DQN"):
            self.train_op = tensorflow.compat.v1.train.AdamOptimizer(self.lr).minimize(loss=self.loss, var_list=self.nn_params)

        with tensorflow.compat.v1.variable_scope("soft_update"):
            TAU = self.update_target_tau
            self.update_op = [tensorflow.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.nn_params_, self.nn_params)]

    def choose_action(self, sess, observation):
        # Explore or Exploit
        explore_p = self.epsilon  # exploration probability

        if np.random.uniform() <= explore_p:
            # Explore: make a random action
            action = np.random.randint(0, self.a_dim)
        else:
            # Exploit: Get action from Q-network
            observation = np.reshape(observation, (1, self.s_dim))
            Qs = sess.run(self.q, feed_dict={self.s: observation})  # shape (1, a_dim)
            action = np.argmax(Qs[0])
        return action

    def learn_a_batch(self, sess):
        # update target every C learning steps
        if self.learn_step_counter % self.update_target_C == 0:
            sess.run(self.update_op)

        # Sample a batch
        s_batch, a_batch, r_batch, done_batch, s2_batch = self.replay_buffer.sample_batch(self.minibatch_size)

        # Train
        _, _, Qhat, loss = sess.run([self.train_op, self.Qa_op, self.q_at_a, self.loss], feed_dict={
            self.s: s_batch, self.a: a_batch, self.r: r_batch, self.done: done_batch, self.s_: s2_batch})

        # count learning steps
        self.learn_step_counter += 1

        # decay exploration probability after each learning step
        if self.epsilon > self.epsilon_stop:
            self.epsilon *= self.epsilon_decay

        return np.max(Qhat)

args = {"env": 'CartPole-v0',
        "random_seed": 1234,
        "max_episodes": 150,  # number of episodes
        "max_episode_len": 200,  # time steps per episode, 200 for CartPole-v0
        ## NN params
        "h1": 32,  # 32
        "h2": 64,  # 64
        "learning_rate": 0.001,  # 1e-3
        "gamma": 0.9,  # 0.9 (32), 0.95 (34) better than 0.99
        "update_target_C": 1,  # update every C learning steps (C=1 if soft update, C=100 if hard update)
        "update_target_tau": 8e-2,  # soft update (tau=8e-2), hard update (tau=1)
        ## exploration prob
        "epsilon_start": 1.0,
        "epsilon_stop": 0.01,  # 0.01
        "epsilon_decay": 0.999,  # 0.999
        ## replay buffer
        "buffer_size": 1e5,
        "minibatch_size": 32,  # 32
        ## tensorboard logs
        "summary_dir": './results/dqn',
        }


def my_sma(x, N):
    """simple moving average over a window of N samples"""
    filt = np.ones(N) / N
    xm = np.convolve(x, filt)
    xm = xm[:-(N-1)]  # remove the last (N-1) elements
    return xm


sess = tensorflow.InteractiveSession()
tensorflow.set_random_seed(int(args['random_seed']))

# initialize numpy seed
np.random.seed(int(args['random_seed']))

# initialize gym env
env = gym.make(args['env'])
env.seed(int(args['random_seed']))
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("states:", env.observation_space)
print("actions:", env.action_space)

# initialize DQN agent
agent = DeepQNetwork(sess, action_size, state_size, args)

# initialize summary (for visualization in tensorboard)
summary_op, ph_reward, ph_Qmax = build_summaries()
subdir = time.strftime("%Y%m%d-%H%M%S", time.localtime())  # a sub folder, e.g., yyyymmdd-HHMMSS
logdir = args['summary_dir'] + '/' + subdir
writer = tensorflow.compat.v1.summary.FileWriter(logdir, sess.graph)  # must be done after graph is constructed

# initialize variables existed in the graph
sess.run(tensorflow.global_variables_initializer())

# training DQN agent
rewards_list = []
loss = -999
num_ep = args['max_episodes']
max_t = args['max_episode_len']
for ep in range(num_ep):
    state = env.reset()  # shape (s_dim,)
    ep_reward = 0  # total reward per episode
    ep_qmax = 0
    t_step = 0
    done = False
    while (t_step < max_t) and (not done):

        # choose an action
        action = agent.choose_action(sess, state)

        # interact with the env
        next_state, reward, done, _ = env.step(action)

        # add the experience to replay buffer
        agent.replay_buffer.add(state, action, reward, done, next_state)

        # learn from a batch of experiences
        if len(agent.replay_buffer) > 3 * agent.minibatch_size:
            qmax = agent.learn_a_batch(sess)
            ep_qmax = max(ep_qmax, qmax)

        # next time step
        t_step += 1
        ep_reward += reward
        state = next_state

    # end of an episode
    rewards_list.append((ep, ep_reward))

    # write to tensorboard summary
    summary_str = sess.run(summary_op, feed_dict={ph_reward: ep_reward, ph_Qmax: ep_qmax})
    writer.add_summary(summary_str, ep)
    writer.flush()

    if ep % 10 == 0:
        print("episode: {}/{}, steps: {}, explore_prob: {:.2f}, total reward: {}".format(ep, num_ep, t_step, agent.epsilon, ep_reward))


eps, rewards = np.array(rewards_list).T

# plot reward v.s. episode
plt.plot(eps, rewards)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()

# check solved requirements
N = 100
thr = 195.0
ep_solve = np.argwhere(my_sma(rewards, N) >= thr).ravel()[0] - N # find where sma > thr
print("episodes before solving: {}".format(ep_solve))


observation_samples = []
# sfeh new code
for ep in range(30):
    state = env.reset()  # shape (s_dim,)
    ep_reward = 0  # total reward per episode
    ep_qmax = 0
    t_step = 0
    done = False
    while (t_step < max_t) and (not done):

        # choose an action
        action = agent.choose_action(sess, state)

        # for this state, an action was chosen
        observation_samples.append([state, action])

        # interact with the env
        state, reward, done, _ = env.step(action)

samples_csv_ready = [['observation0:float', 'observation1:float', 'observation2:float', 'observation3:float', 'action0:float']]
for row in observation_samples:
    samples_csv_ready.append([row[0][0], row[0][1], row[0][2], row[0][3], row[1]])

file_csv = Path('samples.csv')
with open(file_csv, 'w+', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(samples_csv_ready)
csvFile.close()
