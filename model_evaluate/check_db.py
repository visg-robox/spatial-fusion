
import sys
sys.path.append("../")
import os
import common
import shelve
from model_evaluate.eval_ssnet import eval_spnet

MODEL_DIR = common.test_model_path


def delete_keys(model_dir_path):
    # common.make_path(model_dir_path)
    test_state_db_name = os.path.join(model_dir_path, 'current_test_state.db')
    model_test_state = shelve.open(test_state_db_name, flag='c', writeback=True)
    model_name_list = common.find_file_with_pattern('pkl', model_dir_path)
    model_name_list.sort()
    keys = list(model_name_list.keys())
    print("now we have following keys:")
    print('[%s]' % ', '.join(map(str, keys)))
    for key in sys.stdin:
        model_test_state.pop(key)
    print(list)


if __name__ == '__main__':
    delete_keys(MODEL_DIR)




