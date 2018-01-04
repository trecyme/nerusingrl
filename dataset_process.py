import time
import codecs
import warnings
import os
import util

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import numpy as np

'''
    加载Google的词向量，例如'./word2vec/GoogleNews-vectors-negative300.bin'

    @:return model gensim.KeyedVectors对象，可以当成普通dict，也可调用其方法
'''
def load_google_word2vec(path):
    # Load Google's pre-trained Word2Vec model.
    if path:
        print('[' + util.now_time() + '] 开始加载google word2wec词向量:"' + path + '"...')
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        print('[' + util.now_time() + '] 加载结束')
    else:
        model = None
    return model


'''
    加载Glove的词向量，./word2vec/glove.6B.300d.txt

    @:return 词向量（dict） 
'''
def load_glove_word2vec(path):
    if path:
        print('[' + util.now_time() + '] 开始加载glove word2wec词向量:"' + path + '"...')
        word_vectors = {}
        file = codecs.open(path, "r", "UTF-8")
        line_count = 0
        for line in file:
            line = line.split(' ')
            token = line[0]
            if token is None:
                continue
            else:
                word_vectors[token] = np.array([float(x) for x in line[1:]])

            line_count += 1
            # if line_count > 100:
            #     break
        print('[' + util.now_time() + '] 加载结束')
        return word_vectors
    else:
        return {}

'''
    从指定path，加载conll格式的train/valid/test数据集
    例如，加载./conll2003/en/eng.testa，解析成若干行，每行由词及其标签等组成，也就是说无法判断词在文章句子的位置
'''
def load_conll_dataset_tokens(path):

    label_types = []
    all_lines = []
    textfile_count = 0
    try:
        bindex = path.rindex('/')
    except:
        bindex = 0
    textfile_prefix = path[bindex:]
    if path:
        print('[' + util.now_time() + '] 开始加载conll数据集:"' + path + '"...')
        file = codecs.open(path, "r", "UTF-8")
        textfile_name = ''

        for line in file:
            line = line.strip().split(' ')
            newline = []
            if '-DOCSTART-' in line[0]:  # 新的文档
                textfile_count += 1
                textfile_name = (textfile_prefix + "%d") % textfile_count
            elif len(line) == 0 or len(line[0]) == 0:  # 新的句子
                newline.append('')
                all_lines.append(newline)
            else:  # 句子的词和标签
                if line[3] not in label_types:
                    label_types.append(line[3])

                newline.append(line[0])
                newline.append(textfile_name)
                newline.append(0)
                newline.append(0)
                newline.append(line[1])
                newline.append(line[2])
                newline.append(line[3])
                all_lines.append(newline)

        # self.dataset = dataset
        print('[' + util.now_time() + '] 加载结束')
        return all_lines, label_types
