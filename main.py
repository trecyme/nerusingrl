import util
import numpy as np
from DQN import DQN

from environment import Env


def experiment1_train(trainset_path='../dataset/conll2003/en/eng.train',
                      validset_path='../dataset/conll2003/en/eng.testa',
                      testset_path='../dataset/conll2003/en/eng.testb',
                      embedding_path='../dataset/word2vec/glove.6B.300d.txt'):
    n_episodes = 300

    # 初始化环境
    print('[' + util.now_time() + "] init environment...")
    env = Env(trainset_path, embedding_path)
    print('[' + util.now_time() + "] 环境初始化完毕")

    # 初始化DQN
    print('[' + util.now_time() + "] init agent...")
    agent = DQN(n_actions=env.n_actions,status_dim=env.status_dim, action_dim=env.action_dim, reward_dim=env.reward_dim)
    print('[' + util.now_time() + "] agent初始化完毕")

    # 迭代episodes
    for i in range(n_episodes):
        print('[' + util.now_time() + "] start  episode %d of learning" % (i + 1))
        step = 0
        s = env.reset()

        while True:
            # check task is ended
            if env.end():
                break

            # Choose Action a
            a = agent.choose_action(s)

            # Execute action
            s_, r = env.step(a)

            agent.store_transition(s, a, r, s_)

            step += 1
            s = s_

            if step > 200 and step % 5 == 0:
                agent.learn()

    agent.eval_network.save('ex1_eval_model', overwrite=True)


    # def experiment1_test(testset_path='../dataset/conll2003/en/eng.testb',
    #                      embedding_path='../dataset/word2vec/glove.6B.300d.txt'):
    #     # 开始测试
    #     print('[' + util.now_time() + "] 开始测试................................................................")
    #     # 初始化环境
    #     print('[' + util.now_time() + "] init environment...")
    #     env = Env(trainset_path, embedding_path)
    #     print('[' + util.now_time() + "] 环境初始化完毕")
    #
    #     # 初始化DQN
    #     print('[' + util.now_time() + "] init agent...")
    #     agent = DQN(n_actions=env.n_actions, status_dim=env.status_dim, action_dim=env.action_dim,
    #                 reward_dim=env.reward_dim)
    #     print('[' + util.now_time() + "] agent初始化完毕")
    #
    #
    #     print('[' + util.now_time() + "] start  episode %d of learning" % (i + 1))
    #     step = 0
    #     s = env.reset()
    #
    #     while True:
    #         # check task is ended
    #         if env.end():
    #             break
    #
    #         # Choose Action a
    #         a = agent.choose_action(s)
    #
    #         # Execute action
    #         s_, r = env.step(a)
    #
    #         agent.store_transition(s, a, r, s_)
    #
    #         step += 1
    #         s = s_


if __name__ == "__main__":
    experiment1_train()





