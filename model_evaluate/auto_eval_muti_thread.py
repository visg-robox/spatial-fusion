import sys
sys.path.append("../")
import os
import threading
import time

TIME_INTERVAL = 1


class MyThread (threading.Thread):
    def __init__(self, thread_id, command, gpu_space_limit):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.command = command
        self.gpu_space_limit = gpu_space_limit

    def run(self):
        print("Starting thread: " + str(self.thread_id))
        realization(self.command, self.gpu_space_limit)
        print("Exiting thread: " + str(self.thread_id))


def realization(command, gpu_space_limit):
    cur_time = time.time()
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >' + str(cur_time))
    memory_gpu = [int(x.split()[2]) for x in open(str(cur_time), 'r').readlines()]
    # print(memory_gpu)
    for i in range(len(memory_gpu)):
        if memory_gpu[i] >= gpu_space_limit:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
            os.system('rm ' + str(cur_time))
            # os.system(command)
            break


def thread_allocate(max_thread_num, command, gpu_space_limit):
    for i in range(max_thread_num):
        cur_thread = MyThread(i, command, gpu_space_limit)
        cur_thread.start()
        run_with_time_interval()


def run_with_time_interval():
    print(time.time())
    threading.Timer(TIME_INTERVAL, run_with_time_interval).start()


if __name__ == '__main__':
    command = 'python '
    thread_allocate(5, None, 2000)
