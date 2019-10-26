import os
import threading as thd
import time

TIME_INTERVAL = 200
gpu_space = 7000

def run_with_time():
    print(time.time())
    auto_run()
    thd.Timer(TIME_INTERVAL, run_with_time).start()


def auto_run():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    #print(memory_gpu)
    for i in range(len(memory_gpu)):
        if memory_gpu[i] >= gpu_space:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
            os.system('rm tmp')
            os.system("python ./train/feature/STFNet.py ./config/wk/apollo/multi_sequence_record002.txt test spfnet")
            os.system("python ./eval_model/auto_eval.py ./config/wk/apollo/multi_sequence_record002.txt test spfnet")
            break


if __name__ == '__main__':
    run_with_time()

