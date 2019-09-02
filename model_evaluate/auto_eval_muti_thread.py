import sys
sys.path.append("../")
import os
import threading
import time

TIME_INTERVAL = 1

class childThread(threading.Thread):
    def __init__(self, thread_id, gpu_space_limit):
        self.cur_time = thread_id
        self.gpu_space_limit = gpu_space_limit
        self.gpu_state = False

    def run(self):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >' + str(self.cur_time))
        memory_gpu = [int(x.split()[2]) for x in open(str(self.cur_time), 'r').readlines()]
        # print(memory_gpu)
        for i in range(len(memory_gpu)):
            if memory_gpu[i] >= self.gpu_space_limit:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
                os.system('rm ' + str(self.cur_time))
                # os.system(command)
                break

class MyThread (threading.Thread):
    def __init__(self, thread_id, command, gpu_space_limit):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.command = command
        self.gpu_space_limit = gpu_space_limit

    def run(self):
        print("Starting thread: " + str(self.thread_id))
        self.realization(self.command, self.gpu_space_limit)
        print("Exiting thread: " + str(self.thread_id))


    def realization(self, command, gpu_space_limit):
        time_now = time.time()
        child = childThread(time_now, gpu_space_limit)
        child.join()
        os.system(command)


def thread_allocate(max_thread_num, command, gpu_space_limit):
    for i in range(max_thread_num):
        cur_thread = MyThread(i, command, gpu_space_limit)
        cur_thread.start()


if __name__ == '__main__':
    command = 'python '
    thread_allocate(5, None, 2000)
