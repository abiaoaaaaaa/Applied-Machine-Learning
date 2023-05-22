import datetime
from .mk_file import create_folder
import os

# Basic logging class
class Log():
    
    def __init__(self, log_path, file_name):
        self.log_path = log_path
        self.file_name = file_name

    # Create log file
    def open_log_file(self):
        now = datetime.datetime.now()
        file_name = self.file_name+".txt"
        file = open(self.log_path+"/"+file_name,'a+',encoding='utf-8')
        self.file = file
        self.file_name = self.log_path+file_name
    
    # Close log file
    def close_log_file(self):
        self.file.close()

    # Print log
    def log(self, msg='', print_msg=True, end='\n'):
        if print_msg:
            print(msg)
        now = datetime.datetime.now()
        t = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' + str(now.hour).zfill(2) + ':' + str(now.minute).zfill(2) + ':' + str(now.second).zfill(2)
        if isinstance(msg, str):
          lines = msg.split('\n')
        else:
          lines = [msg]
        for line in lines:
          if line == lines[-1]:
            self.file.write('['+t+']'+str(line)+end)
          else:
            self.file.write('['+t+']'+str(line))