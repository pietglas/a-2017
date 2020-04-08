#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import timeit

# write a function that takes a positive integer N, and returns N iff N is prime
def isprime2(n):
    i = 2
    while i <= n/2:
        x = n/i
        if x.is_integer():
            return 0
        i = i + 1
    return n

# Function that creates a list with all the prime numbers less than its argument n
def listofprimes2(n):
    myprimes2 = [isprime2(i) for i in range(1, n, 1) if isprime2(i) == i]
    return myprimes2

# write a function that takes a positive integer N, and returns N iff N is prime.
# It is more than twice as fast as the previous function
def isprime(n):
    if n == 2:
        return 2
    elif (n/2).is_integer():
        return 0
    else:
        i = 3
        while i <= n/3:
            x = n/i
            if x.is_integer():
                return 0
            i += 2
        return n

# function that creates a list with all the prime numbers less than its argument n
def listofprimes(n):
    myprimes = [isprime(i) for i in range(1, n, 1) if isprime(i) == i]
    return myprimes

# test the performance measured in time of both functions. Because we don't input all the arguments, we
# must use named arguments. 
print(timeit.timeit("listofprimes(1000)", number=100, globals=globals()))
print(timeit.timeit("listofprimes2(1000)", number=100, globals=globals()))


