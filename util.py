import time

'''
    Get formatted time
'''
def now_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())