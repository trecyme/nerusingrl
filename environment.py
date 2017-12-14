import dataset_process as dp


class Env:
    def __init__(self, trainset_path, embedding_path):

        # 加载训练集，得到labels类型的数量
        conll_dataset = dp.ConllDataset(trainset_path)
        self.conll_dataset = conll_dataset
        self.action_space = conll_dataset.label_types
        self.n_actions = len(self.action_space)
        self.actions_to_indices = {}
        for action, index in zip(self.action_space, range(self.n_actions)):
            self.actions_to_indices[action] = index

        # 加载词向量
        self.model = dp.load_glove_word2vec(embedding_path)
        self.status_generator = self._status_generator()

        self.status = []
        self.status_dim = 300
        self.reward_dim = 1
        self.action_dim = 1
        self.right_action = 0
        self.done = False

    def reset(self):
        self.status_generator = self._status_generator()
        self.status, self.right_action, self.done = self.status_generator.__next__()
        return self.status

    def end(self):
        return self.done

    def step(self, action):
        # determine reward according to action of agent
        if action != self.right_action:
            reward = -1
        else:
            reward = 1
        self.status, self.right_action, self.done = self.status_generator.__next__()
        return self.status, reward

    def _status_generator(self):
        done = False
        for doc in self.conll_dataset.dataset:
            for sent in doc:
                for word_label in sent:
                    word = []
                    label = None
                    for k in word_label.keys():
                        word = k
                        label = word_label[k]
                    try:
                        status = self.model[word]
                        label_index = self.actions_to_indices[label]
                        yield status, label_index, done
                    except KeyError:
                        # 跳过UNKNOWN word，即在已训练词向量中没有的
                        continue
        yield None, None, True
