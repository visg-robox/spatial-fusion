import time
from threading import Thread


class StopThread:
    def __init__(self):
        self._flag = True

    def terminate(self):
        self._flag = False

    def run(self, n):
        while self._flag and n > 0:
            print('num>>:', n)
            n -= 1
            time.sleep(1)


obj = StopThread()
t = Thread(target=obj.run, args=(11,))
t.start()

time.sleep(5)  # 表示do something

obj.terminate()  # 终止线程
t.join()
print("主线程结束")

