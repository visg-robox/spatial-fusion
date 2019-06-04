
import sys
sys.path.append("../")
import os
import common
import shelve
import threading as thd
import time
from evaluate.eval_ssnet import eval_spnet

MODEL_DIR = common.res_save_path

TIME_INTERVAL = 100


def finder_file(pattern, path='.'):
    matches = []
    dirs = []
    res = []
    for x in os.listdir(path):
        nd = os.path.join(path, x)
        if os.path.isdir(nd):
            dirs.append(nd)
        elif os.path.isfile(nd) and pattern in x:
            matches.append(nd)
    for match in matches:
        res.append(match)
    for dir in dirs:
        res = res + finder_file(pattern, path=dir)
    return res


def run_with_time():
    print(time.time())
    auto_eval(MODEL_DIR)
    thd.Timer(TIME_INTERVAL, run_with_time).start()


def auto_eval(model_dir_path):
    common.make_path(model_dir_path)
    test_state_db_name = os.path.join(model_dir_path, 'current_test_state.db')
    model_test_state = shelve.open(test_state_db_name, flag='c', writeback=True)
    model_name_list = finder_file('pkl', model_dir_path)
    model_name_list.sort()
    for model_name in model_name_list:
        if model_name not in model_test_state:
            model_test_state[model_name] = 1
            eval_spnet(model_name)
    model_test_state.close()


if __name__ == '__main__':
    run_with_time()




