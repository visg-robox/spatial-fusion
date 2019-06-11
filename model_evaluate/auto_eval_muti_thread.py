import sys
sys.path.append("../")
import os
import threading
import time

TIME_INTERVAL = 1
GPU_SPACE_LIMIT = 2000


def check_gpu(gpu_space_limit):
    cur_time = time.time()
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >' + str(cur_time))
    memory_gpu = [int(x.split()[2]) for x in open(str(cur_time), 'r').readlines()]
    # print(memory_gpu)
    for i in range(len(memory_gpu)):
        if memory_gpu[i] >= gpu_space_limit:
            os.system('rm ' + str(cur_time))
            return i
    return None


class MyThread (threading.Thread):
    def __init__(self, thread_id, command, gpu_space_limit):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.command = command
        self.gpu_space_limit = gpu_space_limit

    def run(self):
        while 1:
            available_gpu = check_gpu(GPU_SPACE_LIMIT)
            if available_gpu is not None:
                break
            time.sleep(5)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpu)
        os.environ(self.command)


def auto_run():
    pass




def thread_allocate(max_thread_num, command, gpu_space_limit):
    for i in range(max_thread_num):
        cur_thread = MyThread(i, command, gpu_space_limit)
        cur_thread.start()


def check_gpu_with_time_interval():
    print(time.time())
    if check_gpu(GPU_SPACE_LIMIT):
        return
    threading.Timer(TIME_INTERVAL, check_gpu_with_time_interval).start()


def func():
    print("hello, world")


if __name__ == '__main__':
    command = 'python '
    # thread_allocate(5, None, 2000)
