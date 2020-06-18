from multiprocessing import Pool
import logging

def aaa(a):
    print(a)


def multiproc_version():
    with Pool(5) as p:
        input = [12, 12, 23, 34]
        p.map(aaa, input)

def sequential_version():
    input = [12, 12, 23, 34]
    for a in input:
        aaa(a)

if __name__ == "__main__":
    sequential_version()


