
import sys
sys.path.append("../")
import os
import common
import shelve
from model_evaluate.eval_ssnet import eval_spnet

MODEL_DIR = common.test_model_path


def find_list_with_pattern(pattern, input_list):
    res = None
    for x in input_list:
        if pattern in x:
            res = x
    return res


def delete_keys(model_dir_path):
    # common.make_path(model_dir_path)
    test_state_db_name = os.path.join(model_dir_path, 'current_test_state.db')
    model_test_state = shelve.open(test_state_db_name, flag='c', writeback=True)
    db_keys = list(model_test_state.keys())
    db_keys.sort()
    print("now we have following keys:")
    print('%s' % ',\n '.join(map(str, db_keys)))
    print('-----------------------------')
    print("enter the step number you want to delete:")
    for line in sys.stdin:
        list_new = line.split()
        for step in list_new:
            delete_key = find_list_with_pattern(step, db_keys)
            model_test_state.pop(delete_key)
            print('you delete: ' + delete_key)
    print(list)


if __name__ == '__main__':
    delete_keys(MODEL_DIR)




