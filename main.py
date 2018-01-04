import util

import sys
import os
import evaluate

from environment import Env
from DQN import DQN
import dataset_process as dp

def experiment1_train(output_folder,
                      word_vectors,
                      n_episodes=300,
                      trainset_path='./dataset/conll2003/en/eng.train',
                      ):
    # 初始化环境
    print('[' + util.now_time() + "] init environment...")
    env = Env(trainset_path, word_vectors)
    print('[' + util.now_time() + "] 环境初始化完毕")

    # 初始化DQN
    print('[' + util.now_time() + "] init agent...")
    agent = DQN(n_actions=env.n_actions, status_dim=env.status_dim, action_dim=env.action_dim,
                reward_dim=env.reward_dim)
    print('[' + util.now_time() + "] agent初始化完毕")

    # 迭代episodes
    for i in range(n_episodes):
        print('[' + util.now_time() + "] start episode %03d of learning..." % (i + 1))
        step = 0
        s = env.reset()

        while True:
            # check task is ended
            if env.end():
                print('[' + util.now_time() + "] episode %03d of learning...done" % (i + 1))
                result_file = '%03d_episode_train.txt' % (i + 1)
                env.save_all_newlines_to_file(output_folder, result_file)
                train_eval = evaluate.conlleval(output_folder, result_file)
                test_eval = experiment1_test(output_folder, word_vectors, agent, i)
                break

            # Choose Action a
            a = agent.choose_action(s)

            # Execute action
            # print('step %d' % step)
            s_, r = env.step(a)

            agent.store_transition(s, a, r, s_)

            step += 1
            s = s_

            if step > 200 and step % 5 == 0:
                agent.learn()

    # plot and compare train and test set TODO
    # plot(train_evals,test_evals)
    agent.eval_network.save(output_folder+os.path.sep+'ex1_eval_model', overwrite=True)


def experiment1_test(output_folder,
                     word_vectors,
                     agent,
                     episode_index,
                     testset_path='./dataset/conll2003/en/eng.testb',
                     ):
    # 初始化环境
    env = Env(testset_path, word_vectors)
    step = 0
    s = env.reset()
    print('[' + util.now_time() + "] start testing...")
    while True:
        # check task is ended
        if env.end():
            print('[' + util.now_time() + "] testing...done")
            result_file = '%03d_episode_test.txt' % (episode_index + 1)
            env.save_all_newlines_to_file(output_folder, result_file)
            return evaluate.conlleval(output_folder, result_file)

        # Choose Action a
        a = agent.choose_action(s)

        # Execute action
        s_, r = env.step(a)

        # Next status
        step += 1
        s = s_


if __name__ == "__main__":
    # 确定实验名称，和输出文件夹
    output_folder = './output'
    experiment_name = 'ex1_training_' + util.now_date()
    args = sys.argv[1:]
    if len(args) > 0 and args[0]:  # 如果传入参数，第一个参数则为实验名
        experiment_name = args[0]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_folder = output_folder + os.path.sep + experiment_name
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # 初始化词向量，作为全局变量
    # embeding_path = './dataset/word2vec/GoogleNews-vectors-negative300.bin.gz'
    embeding_path = './dataset/word2vec/glove.6B.300d.txt'
    print('[' + util.now_time() + "] 初始化词向量...")
    word_vectors = dp.load_glove_word2vec(embeding_path)
    print('[' + util.now_time() + "] 词向量初始化完毕")

    # 训练模型
    experiment1_train(output_folder, word_vectors, n_episodes=3)


