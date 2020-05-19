import numpy as np
import random
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import utils as np_utils
 
tf.compat.v1.disable_eager_execution()

class ActorCritic():

    def __init__(self, env):
        
        # hyperparameters
        self.env = env
        self.epochs = 1
        self.gamma = 0.99
        self.actor_learning_rate = 0.01
        self.critic_learning_rate = 0.00001
        self.n_s = env.observation_space.shape[0]
        self.n_a = env.action_space.n
        self.description = 'Actor-Critic'
        self.verbose = False
        self.memory = [] 
        self.nn_batch_size = None
        self.minibatch_sz = 64

        # memory
        self.memory_max = 50000

        self.reset()


    def reset(self):
            
        # Initialize actor model
        model = Sequential()
        model.add(Dense(64, input_dim=self.n_s, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_a, activation = 'softmax'))
        self.model = model
        self._build_train_fn()

        # Initialize critic advantage model
        critic = Sequential()
        critic.add(Dense(32, input_dim=self.n_s, activation='relu'))
        critic.add(Dense(32, activation='relu'))
        critic.add(Dense(self.n_a, activation='relu'))
        critic.compile(loss='mse', optimizer=Adam(lr=self.critic_learning_rate))
        self.critic = critic

        self.reset_experience()

    def reset_experience(self):

        self.experience_cache = {
            's': [],
            'a': [],
            'r': [],
        }

    def pick_action(self, state):

        tmp = self.model.predict(np.array([state,]))
        action = np.random.choice(np.arange(self.n_a), p=tmp[0])
        
        if self.verbose: 
            print(f'prediction confidence {tmp}')

        return action


    def _build_train_fn(self):

        """
        Adapted from https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
        """

        action_prob_placeholder = self.model.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.n_a), name="action_onehot")
        q_placeholder = K.placeholder(shape=(None,), name="q_values")

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * q_placeholder
        loss = K.mean(loss)

        adam = Adam(lr=self.actor_learning_rate)

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           q_placeholder],
                                   outputs=[loss],
                                   updates=updates)


    def _batch_train_(self):

        if len(self.memory) > self.minibatch_sz:

            # create training batch
            batch = random.sample(self.memory, self.minibatch_sz)

            # get predictions
            s_vec = np.array([x[0] for x in batch])
            sp_vec = np.array([x[3] for x in batch])

            m_pred = self.critic.predict(s_vec)
            tm_pred = self.critic.predict(sp_vec)

            # use update rule from Minh 2013
            for i in range(len(batch)): 
                s, a, r, s_prime, done = batch[i]
                target = r
                if not done:

                    target += self.gamma * np.max(tm_pred[i])
                m_pred[i][a] = target

            self.critic.fit(s_vec, m_pred, epochs = self.epochs, 
                                          batch_size = self.nn_batch_size,
                                          verbose = False)

    def update(self, s, a, r, s_prime, done):

        self.experience_cache['s'].append(s)
        self.experience_cache['a'].append(a)
        self.experience_cache['r'].append(r)
        
        if done:

            states = np.array(self.experience_cache['s'])
            actions = np.array(self.experience_cache['a'])
            rewards = np.array(self.experience_cache['r'])

            # One-hot encode the actions
            action_onehot_encoded =  np_utils.to_categorical(actions, num_classes=self.n_a)

            # Get predicted Q-values
            q_pred = self.critic.predict(states)
            q_values = np.zeros(len(states))
            for i in range(len(q_values)):
                q_values[i] = q_pred[i][actions[i]]

            # Train actor
            loss = self.train_fn([
                states,
                action_onehot_encoded,
                q_values
            ])

            # Train critic
            if len(self.memory) > self.memory_max: self.memory.pop(0)
            self.memory.append([s, a, r, s_prime, done])
            self._batch_train_()


            