from multiprocessing import Process
from multiprocessing import Manager
import concurrent.futures
from multiprocessing import Queue
import os

import math

from sklearn.svm import l1_min_c




def make_cal_one(i, a, b, c, return_dict):
    print(a,b,c)

    return_dict[i] = i 
    #return sum(_list), len(_list), len(_list), len(_list), 

def make_cal(numbers, a, b, c):
    print(a,b,c)
    _list = []
    for number in numbers:
        _list.append(math.sqrt(number ** 3 + a*b*c))
    

if __name__ == '__main__':
    
    manager = Manager()
    return_dict = manager.dict()

    jobs = [None]*5
    for i in range(5):
        jobs[i] = Process(target=make_cal_one, args=(i, i, i+2, i+3, return_dict))
        jobs[i].start()

    for proc in jobs:
        proc.join()

    print(return_dict.values())

  