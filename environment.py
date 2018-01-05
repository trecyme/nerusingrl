import dataset_process as dp
import os
import codecs
import util


class Env:
    def __init__(self, dataset_path, word_vectors):

        # 加载训练集，得到labels类型的数量
        self.conll_lines, self.actions_labels = dp.load_conll_dataset_tokens(dataset_path)
        self.line_count = -1 # 用于统计现在的 状态处于conll_lines的哪一行
        self.n_actions = len(self.actions_labels)
        self.labels_actions = {}
        for label, index in zip(self.actions_labels, range(self.n_actions)):
            self.labels_actions[label] = index

        self.status_generator = self._status_generator()

        self.word_vectors = word_vectors
        self.status = []
        self.status_dim = 300
        self.reward_dim = 1
        self.action_dim = 1
        self.actual_action = 0
        self.done = False

    def reset(self):
        self.done = False
        self.status = []
        self.line_count = -1
        self.status_generator = self._status_generator()
        self.status, self.actual_action, self.done = self.status_generator.__next__()
        return self.status

    def end(self):
        return self.done

    def step(self, action):
        # determine reward according to action of agent
        if action != self.actual_action:
            reward = -1
        else:
            reward = 1

        # record predicted action of this status, for later evaluate
        if 0 <= self.line_count < len(self.conll_lines):
            if len(self.conll_lines[self.line_count]) >= 8:                         # cover
                self.conll_lines[self.line_count][7] = self.actions_labels[action]
            else:                                                                   # first episode of predicting
                self.conll_lines[self.line_count].append(self.actions_labels[action])
            # print('append or cover: %s' % self.conll_lines[self.line_count][7])
        else:
            print('[' + util.now_time() + "] ERROR: line count error!")

        # after execute this action, env step into next status
        self.status, self.actual_action, self.done = self.status_generator.__next__() # 执行操作后的状态

        return self.status, reward

    def save_all_newlines_to_file(self, path, file):
        print('[' + util.now_time() + "] save_all_newlines_to_file...")
        if path and not os.path.exists(path):
            os.makedirs(path)

        line_count = 0
        if path and file:
            file = codecs.open(path + os.path.sep + file, "w", encoding="UTF-8")
            for line in self.conll_lines:
                n_tokens = len(line)

                # check every line
                if n_tokens == 0 or n_tokens == 1:
                    # file.write("\n")
                    continue
                elif n_tokens == 7:                      # UNK word has been skipped, so it has no preditced label, but in fact 'O'
                    # print(line)
                    line.append('O')
                elif n_tokens == 8:
                    pass
                else:
                    print('[' + util.now_time() + '] Unexpected: '+line)
                    continue

                # write to file
                n_tokens = len(line)
                for index in range(n_tokens - 1):
                    file.write("%s " % line[index])
                file.write(line[-1] + "\n")

                # print for debug
                # if line_count % 10 == 0:
                #     print('line len: %d' % n_tokens)
            file.close()
            print('[' + util.now_time() + "] save_all_newlines_to_file...done")

    def _status_generator(self):
        done = False
        unk_count = 1
        for line in self.conll_lines:
            self.line_count += 1
            if line is not None and len(line) != 0 and len(line[0]) != 0:
                word = line[0]
                label = line[6]
                try:
                    status = self.word_vectors[word]
                except KeyError:
                    try:
                        status = self.word_vectors[word.lower()]            # transform word to lower case
                    except KeyError:
                        # 跳过UNKNOWN word，即在已训练词向量中没有的
                        # print("UNK -- {0} and {1}".format(word,word.lower()))
                        unk_count += 1
                        continue
                actual_action = self.labels_actions[label]
                yield status, actual_action, done
            else:
                continue
        print('[' + util.now_time() + "] UNK word count: %d" % unk_count)
        yield None, None, True

class Env1(Env):
    def __init__(self, dataset_path, word_vectors):

        # 加载训练集，得到labels类型的数量
        self.sentences, self.actions_labels = dp.load_conll_dataset_sents(dataset_path)
        self.line_count = -1 # 用于统计现在的状态处于sentence的哪一行
        self.sentence_count = -1 # 用于统计现在的 状态处于sentences的哪一句
        self.n_actions = len(self.actions_labels)
        self.labels_actions = {}
        for label, index in zip(self.actions_labels, range(self.n_actions)):
            self.labels_actions[label] = index

        self.status_generator = self._status_generator()

        self.word_vectors = word_vectors
        self.status = []
        self.status_dim = 300
        self.reward_dim = 1
        self.action_dim = 1
        self.actual_action = 0
        self.done = False

    def reset(self):
        self.done = False
        self.status = []
        self.line_count = -1
        self.sentence_count = -1
        self.status_generator = self._status_generator()
        self.status, self.actual_action, self.done = self.status_generator.__next__()
        return self.status

    def step(self, action):
        # determine reward according to action of agent
        if action != self.actual_action:
            reward = -1
        else:
            reward = 1

        # record predicted action of this status, for later evaluate
        if 0 <= self.sentence_count < len(self.sentences):
            line = self.sentences[self.sentence_count][self.line_count]
            if len(line) >= 8:                         # cover
                self.sentences[self.sentence_count][self.line_count][7] = self.actions_labels[action]
            else:                                                                   # first episode of predicting
                self.sentences[self.sentence_count][self.line_count].append(self.actions_labels[action])
            # print('append or cover: %s' % self.conll_lines[self.line_count][7])
        else:
            print('[' + util.now_time() + "] ERROR: line count error!")

        # after execute this action, env step into next status
        self.status, self.actual_action, self.done = self.status_generator.__next__() # 执行操作后的状态

        return self.status, reward

    def save_all_newlines_to_file(self, path, file):
        print('[' + util.now_time() + "] save_all_newlines_to_file...")
        if path and not os.path.exists(path):
            os.makedirs(path)

        if path and file:
            file = codecs.open(path + os.path.sep + file, "w", encoding="UTF-8")
            for sentence in self.sentences:
                for line in sentence:
                    n_tokens = len(line)

                    # check every line
                    if n_tokens == 0 or n_tokens == 1:
                        # file.write("\n")
                        continue
                    elif n_tokens == 7:                      # UNK word has been skipped, it has no preditced label, but in fact 'O', so append 'O'
                        # print(line)
                        line.append('O')
                    elif n_tokens == 8:
                        pass
                    else:
                        print('[' + util.now_time() + '] Unexpected: '+line)
                        continue

                    # write to file
                    n_tokens = len(line)
                    for index in range(n_tokens - 1):
                        file.write("%s " % line[index])
                    file.write(line[-1] + "\n")
                file.write("\n")
            file.close()
            print('[' + util.now_time() + "] save_all_newlines_to_file...done")

    def _status_generator(self):
        done = False
        unk_count = 1
        for sentence in self.sentences:
            self.sentence_count += 1
            self.line_count = -1
            for line in sentence:
                self.line_count += 1
                if line is not None and len(line) != 0 and len(line[0]) != 0:
                    word = line[0]
                    label = line[6]
                    try:
                        if self.line_count == 0:
                            status = self.word_vectors[word]
                        else:
                            status = self.status + self.word_vectors[word]
                    except KeyError:
                        try:
                            if self.line_count == 0:
                                status = self.word_vectors[word.lower()]
                            else:
                                status = self.status + self.word_vectors[word.lower()]  # transform word to lower case
                        except KeyError:
                            # 跳过UNKNOWN word，即在已训练词向量中没有的
                            # print("UNK -- {0} and {1}".format(word,word.lower()))
                            unk_count += 1
                            continue
                    actual_action = self.labels_actions[label]
                    yield status, actual_action, done
                else:
                    continue

        print('[' + util.now_time() + "] UNK word count: %d" % unk_count)
        yield None, None, True
