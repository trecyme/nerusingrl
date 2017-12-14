from keras.layers import Input, Dense
from keras.models import Model
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np


class DQN:
    def __init__(self,
                 n_actions,
                 status_dim,
                 reward_dim,
                 action_dim,
                 epsilon=0.9,
                 gamma=0.9,
                 alpha=0.1,
                 memory_size=2000,
                 batch_size=32
                 ):

        self.memory_counter = 0
        self.learn_counter = 0
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.memory_size = memory_size
        self.status_dim = status_dim
        self.reward_dim = reward_dim                 # TODO 是否该去掉
        self.action_dim = action_dim                 # TODO 是否该去掉
        self.batch_size = batch_size
        self.memory = np.zeros((self.memory_size, status_dim * 2 + action_dim + reward_dim))
        self.eval_network, self.target_network = self._build_net()
        print('epsilon: %d ' % epsilon)
        print('n_actions: %d ' % n_actions)
        print('gamma: %d ' % gamma)
        # print('alpha: %d ' % alpha)
        print('memory_size: %d ' % memory_size)
        print('status_dim: %d ' % status_dim)
        print('reward_dim: %d ' % reward_dim)
        print('action_dim: %d ' % action_dim)
        print('batch_size: %d ' % batch_size)

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        # random greedy choice action
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.eval_network.predict(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.memory_size
        if s_ is None:
            s_ = np.zeros((self.status_dim))
        try:
            transition = np.concatenate((s, [a, r], s_))
        except:
            print('ERROR:')
            print('s:')
            print(s)
            print('s_:')
            print(s_)
            print('a:')
            print(a)
            print('r:')
            print(r)
            print('-----------------------------------------------------------')
        # if self.memory_counter < self.memory_size:
        #     self.memory.append(transition)
        # else:
        self.memory[index, :] = transition
        self.memory_counter += 1

    def _build_net(self):
        # Eval Network
        eval_model_inputs = Input(shape=(self.status_dim,))
        eval_model_l1 = Dense(64, activation='relu')(eval_model_inputs)
        eval_model_outputs = Dense(self.n_actions, activation='softmax')(eval_model_l1)
        eval_model = Model(inputs=eval_model_inputs, outputs=eval_model_outputs)
        eval_model.compile(optimizer='RMSProp',
                           loss='categorical_crossentropy')

        # copy of old Eval Network
        copy_eval_model_inputs = Input(shape=(self.status_dim,))
        copy_eval_model_l1 = Dense(64, activation='relu')(copy_eval_model_inputs)
        copy_eval_model_outputs = Dense(self.n_actions, activation='softmax')(copy_eval_model_l1)
        copy_eval_model = Model(inputs=copy_eval_model_inputs, outputs=copy_eval_model_outputs)

        return eval_model, copy_eval_model

    def learn(self):
        # replace variables of target network with variables of eval
        self.target_network.set_weights(self.eval_network.get_weights())

        # random pick sample indices of memory of mini batch size
        if self.memory_counter > self.memory_size:
            batch_indices = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            batch_indices = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[batch_indices, :]

        # use two network to predict status's q value
        batch_q_eval = self.eval_network.predict(batch_memory[:, :self.status_dim], self.batch_size,verbose=0)
        batch_q_next = self.target_network.predict(batch_memory[:, -self.status_dim:], self.batch_size,verbose=0)

        # print(type(batch_q_eval))
        # print(type(batch_q_eval))
        # print(type(batch_indices))

        batch_indices = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.status_dim].astype(int)

        reward = batch_memory[:, (self.status_dim + self.action_dim)]

        # print('reward:')
        # print(reward)
        # print('np.max(q_next, axis=1)')
        # print(np.max(batch_q_next, axis=1))

        batch_q_target = batch_q_eval.copy()
        batch_q_target[batch_indices, eval_act_index] = reward + self.gamma * np.max(batch_q_next, axis=1)

        self.eval_network.evaluate(batch_memory[:, :self.status_dim],
                                   batch_q_target,
                                   verbose=0,
                                   batch_size=self.batch_size)
