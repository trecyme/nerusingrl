import time

'''
    Get formatted time
'''
def now_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


'''
    Get formatted data
'''
def now_date():
    return time.strftime("%Y_%m_%d", time.localtime())