import os
import util


def conlleval(output_folder, conll_file):
    print('[' + util.now_time() + '] 开始评估本次效果:"' + conll_file + '"...')

    # CoNLL evaluation script
    conll_evaluation_script = os.path.join('.', 'conlleval')
    conll_eval_file = '{0}_conll_evaluation.txt'.format(conll_file)
    conll_file_path = output_folder + os.path.sep + conll_file
    conll_eval_file_path = output_folder + os.path.sep + conll_eval_file
    shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script, conll_file_path, conll_eval_file_path)
    print('[' + util.now_time() + '] shell_command: {0}'.format(shell_command))
    os.system(shell_command)
    with open(conll_eval_file_path) as f:
        print(f.read())
    print('[' + util.now_time() + "] 评估完成")

