
import sys
sys.path.append("../")
import os
import common
import shelve
import threading as thd
import time
from multiprocessing import set_start_method
from model_evaluate.eval_ssnet import eval_spnet, eval_spnet_multi_process


MODEL_DIR = common.test_model_path

TIME_INTERVAL = 200

try:
    set_start_method('spawn')
except RuntimeError:
    pass

def run_with_time():
    print(time.time())
    auto_eval(MODEL_DIR)
    thd.Timer(TIME_INTERVAL, run_with_time).start()


def auto_eval(model_dir_path):
    '''
    check model in the model_dir_path
    choose the latest model which not tested before
    run eval_spnet_multi_process
    :param model_dir_path:
    :return:
    '''
    #common.make_path(model_dir_path)
    test_state_db_name = os.path.join(model_dir_path, 'current_test_state.db')
    model_test_state = shelve.open(test_state_db_name, flag='c', writeback=True)
    model_name_list = common.find_file_with_pattern('pkl', model_dir_path)
    model_name_list.sort()
    for model_name in model_name_list:
        if model_name not in model_test_state:
            eval_spnet_multi_process(model_name)
            model_test_state[model_name] = 1
    model_test_state.close()


if __name__ == '__main__':
    run_with_time()




