import os
import commands 
import time

if __name__ == '__main__':
    while 1:
        cmd = 'ps aux | grep manage.py'
        status, output = commands.getstatusoutput(cmd)
        cnt = output.count('manage.py')
        if cnt <= 2:
            cmd = 'nohup python ./match/manage.py runserver 0.0.0.0:8003 &'
            print cmd
            status, output = commands.getstatusoutput(cmd)
            print status
            print output
        time.sleep(600)
