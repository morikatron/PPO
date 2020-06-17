import random
from collections import deque

from tqdm import tqdm
import gym
import numpy as np
import tensorflow as tf

from util import set_global_seeds
from ppo import PPO
from config import Config

import logging
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)
sh = StreamHandler()
sh.setLevel(DEBUG)
logger.addHandler(sh)
logging.basicConfig(filename='logs.txt', filemode='w', format='%(message)s', level=DEBUG)


class CartPoleWrapper(gym.Wrapper):
    """
    CartPoleに関しては、元の環境の報酬の与え方が「1ステップごとに棒が立っていたら報酬が1与えられる」となっていますが、これを
    「最後まで立っていたら報酬1, その前に倒れたら報酬-1、途中の報酬は0」という形に変えます。
    """
    def __init__(self, env):
        super(CartPoleWrapper, self).__init__(env)
        self.steps = 0
        version = env.spec.id.split('-')[-1]
        if version == 'v0':
            self._max_step = 200
        else:
            self._max_step = 500

    def reset(self):
        obs = self.env.reset()
        self.steps = 0
        return obs

    def step(self, ac):
        self.steps += 1
        obs, rew, done, info = self.env.step(ac)
        if done:
            if self.steps == self._max_step:
                rew = 1
            else:
                rew = -1
        else:
            rew = 0
        return obs, rew, done, info


class Memory:
    """
    集めたサンプルを学習用に保存しておくメモリークラスです。
    サンプルを収集するステップ数分のサイズを
    deltaの値を計算する際に1ステップ先の状態価値が必要になるため、valueのサイズは1つ余分に大きくとり」、
    GAEを計算する際に1ステップ先のGAEの値が必要になるため、GAEのサイズを1つ余分に大きくとっています。
    """
    def __init__(self, obs_shape, hparams):
        self._hparams = hparams
        self.size = self._hparams.num_step
        self.obses   = np.zeros((self.size, )+obs_shape)
        self.actions = np.zeros((self.size, ))
        self.rewards = np.zeros((self.size,   1))
        self.dones   = np.zeros((self.size,   1))
        self.values  = np.zeros((self.size+1, 1))
        self.policy  = np.zeros((self.size,   1))
        self.deltas  = np.zeros((self.size,   1))
        self.discounted_rew_sum = np.zeros((self.size, 1))
        self.gae = np.zeros((self.size+1, 1))
        self.i = 0  # サンプルをメモリに保存するためのポインタの役割を果たします

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, done, value, policy):
        """
        サンプルをメモリに保存する時点では、 1ステップ先の状態価値はまだ不明なのでdeltaのみ1ステップ分遅延させて保存します。
        その都合上メモリのサイズより1回分多くaddが呼ばれますが、最後のaddではdeltaのみ保存するようにします。
        """
        if self.i < len(self.obses):
            self.obses[self.i] = obs
            self.actions[self.i] = action
            self.rewards[self.i] = reward
            self.dones[self.i] = done
            self.values[self.i] = value
            self.policy[self.i] = policy

        if self.i > 0:
            # ステップtでエピソードが終了していた場合、V(t+1)は次のエピソードの最初の状態価値なのでdeltaを計算する際にエピソードをまたがないようにV(t+1)は0とします。(エピソード終了後に得られる報酬は0なので状態価値も0です。)
            self.deltas[self.i - 1] = self.rewards[self.i - 1] + self._hparams.gamma * value * (1 - self.dones[self.i - 1]) - self.values[self.i - 1]
        self.i += 1

    def compute_gae(self):
        """
        Generalized Advantage Estimatorを後ろからさかのぼって計算します。
        最後の状態のGAEはdeltaと等しく、それ以前は次の状態のgaeをgamma * lambdaで割り引いた値にdeltaの値を足したものになります。
        ただし、ここでもエピソードをまたいで計算しないようにgae
        openAI spinning upやRLlibではscipyのlfilterを使ったgaeの計算が使われていますが、
        :return:
        """
        self.gae[-1] = self.deltas[-1]
        for t in reversed(range(self.size-1)):
            self.gae[t] = self.deltas[t] + (1 - self.dones[t]) * (self._hparams.gamma * self._hparams.lambda_) * self.gae[t + 1]
        self.discounted_rew_sum = self.gae[:-1] + self.values[:-1]
        self.gae = (self.gae - np.mean(self.gae[:-1])) / (np.std(self.gae[:-1]) + 1e-8)  # 正規化をしておきます。
        return

    def sample(self, idxes):
        batch_obs = tf.convert_to_tensor(self.obses[idxes], dtype=tf.float32)
        batch_act = tf.convert_to_tensor(self.actions[idxes], dtype=tf.int32)
        batch_adv = tf.squeeze(tf.convert_to_tensor(self.gae[idxes], dtype=tf.float32))
        batch_pi = tf.squeeze(tf.convert_to_tensor(self.policy[idxes], dtype=tf.float32))
        batch_sum = tf.squeeze(tf.convert_to_tensor(self.discounted_rew_sum[idxes], dtype=tf.float32))
        return batch_obs, batch_act, batch_adv, batch_sum, batch_pi

    def reset(self):
        self.i = 0
        self.obses = np.zeros_like(self.obses)
        self.actions = np.zeros_like(self.actions)
        self.rewards = np.zeros_like(self.rewards)
        self.values = np.zeros_like(self.values)
        self.policy = np.zeros_like(self.policy)
        self.deltas = np.zeros_like(self.deltas)
        self.discounted_rew_sum = np.zeros_like(self.discounted_rew_sum)
        self.gae = np.zeros_like(self.gae)


def main():
    config = Config()
    set_global_seeds(config.seed)
    env = gym.make(config.env_name)
    env = CartPoleWrapper(env)
    # with tf.device("/gpu:0"):  # gpuを使用する場合
    with tf.device("/cpu:0"):
        ppo = PPO(
            num_actions=env.action_space.n,
            input_shape=env.observation_space.shape,
            config=config
        )
    num_episodes = 0
    episode_rewards = deque([0] * 100, maxlen=100)
    memory = Memory(env.observation_space.shape, config)
    reward_sum = 0
    obs = env.reset()
    for t in tqdm(range(config.num_update)):
        # ===== get samples =====
        for _ in range(config.num_step):
            policy, value = ppo.step(obs[np.newaxis, :])
            policy = policy.numpy()
            action = np.random.choice(2, p=policy)
            next_obs, reward, done, _ = env.step(action)
            memory.add(obs, action, reward, done, value, policy[action])
            obs = next_obs
            reward_sum += reward
            if done:
                episode_rewards.append(env.steps)
                num_episodes += 1
                reward_sum = 0
                obs = env.reset()
        _, last_value = ppo.step(obs[np.newaxis, :])
        memory.add(None, None, None, None, last_value, None)

        # ===== make mini-batch and update parameters =====
        memory.compute_gae()
        for _ in range(config.num_epoch):
            idxes = [idx for idx in range(config.num_step)]
            random.shuffle(idxes)
            for start in range(0, len(memory), config.batch_size):
                minibatch_indexes = idxes[start:start+config.batch_size]
                batch_obs, batch_act, batch_adv, batch_sum, batch_pi_old = memory.sample(minibatch_indexes)
                loss, policy_loss, value_loss, entropy_loss, policy, kl, frac = ppo.train(batch_obs, batch_act, batch_pi_old, batch_adv, batch_sum)
        memory.reset()
        if t % config.log_step == 0:
            logger.info("\nnum episodes: {}".format(num_episodes))
            logger.info("loss: {}".format(loss.numpy()))
            logger.info("policy loss: {}".format(policy_loss.numpy()))
            logger.info("value loss: {}".format(value_loss.numpy()))
            logger.info("entropy loss: {}".format(entropy_loss.numpy()))
            logger.info("kl: {}".format(kl.numpy()))
            logger.info("frac: {}".format(frac.numpy()))
            logger.info("mean 100 episode reward: {}".format(np.mean(episode_rewards)))
            logger.info("max 100 episode reward: {}".format(np.max(episode_rewards)))
            logger.info("min 100 episode reward: {}".format(np.min(episode_rewards)))

    # ===== finish training =====
    if config.play:
        obs = env.reset()
        while True:
            action, _ = ppo.step(obs[np.newaxis, :])
            action = int(action.numpy()[0])
            obs, _, done, _ = env.step(action)
            env.render()
            if done:
                obs = env.reset()


if __name__ == "__main__":
    main()
