# this script contains examples of decorator functions. 

import time 
 
def timer(func): 
  """A decorator that prints how long a function took to run."""  
  # Define the wrapper function to return. 
  def wrapper(*args, **kwargs):  
    # When wrapper() is called, get the current time. 
    t_start = time.time()  
    # Call the decorated function and store the result. 
    result = func(*args, **kwargs)  
    # Get the total time it took to run, and print it. 
    t_total = time.time() - t_start 
    print('{} took {}s'.format(func.__name__, t_total))  
    return result  
  return wrapper
# example usage
@timer 
def sleep_n_seconds(n): 
  time.sleep(n) 

n = input('enter a number for timer: ')
n = int(n)

sleep_n_seconds(n)







def memoize(func): 
  """Store the results of the decorated function for fast lookup 
  """  
  # Store results in a dict that maps arguments to results 
  cache = {}  
  # Define the wrapper function to return. 
  def wrapper(*args, **kwargs): 
    # Define a hashable key for 'kwargs'. 
    kwargs_key = tuple(sorted(kwargs.items())) 
    # If these arguments haven't been seen before, 
    if (args, kwargs_key) not in cache: 
      # Call func() and store the result. 
      cache[(args, kwargs_key)] = func(*args, **kwargs)  
    return cache[(args, kwargs_key)]  
  return wrapper 

# example usage
@memoize 
def slow_function(a, b): 
  print('Sleep 5 seconds...') 
  print('1')
  time.sleep(1) 
  print('2')
  time.sleep(1) 
  print('3')
  time.sleep(1) 
  print('4')
  time.sleep(1) 
  print('5')
  time.sleep(1) 
  return a + b 

a = input('enter a number for memoize a: ')
a = int(a)
b = input('enter a number for memoize a: ')
b = int(b)

slow_function(a,b)




def run_three_times(func): 
    """Run the inner function x times"""  
    def wrapper(*args, **kwargs): 
        for i in range(3): 
            func(*args, **kwargs) 
    return wrapper  
  
  
@run_three_times 
def print_sum(a, b):
    print(a + b)  

print_sum(3, 5)

