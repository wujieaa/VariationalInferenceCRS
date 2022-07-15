import os
import time
class LogToFile:
    def __init__(self,log_file_path):
        dir=os.path.dirname(log_file_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.path=log_file_path
    def clear(self):
        with open(self.path, 'w',encoding="utf-8") as f:
            pass
    def log(self,*messages,prefix='time'):
        with open(self.path,'a',encoding="utf-8") as f:
            if prefix=='time':
                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.write(f'-------------{now}----------------\n')
            for msg in messages:
                f.write(msg)
                f.write('\n')


