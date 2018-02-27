import time
from multiprocessing import Process

def long_r_function():
    import rpy2.robjects as robjects
    robjects.r('Sys.sleep(10)')    # pretend we have a long-running function in R

if __name__ == '__main__':
    r_process = Process(target=long_r_function)
    start_time = time.time()
    r_process.start()

    while r_process.is_alive():
        print("R running...")    # expect to be able to see this print from main process,
        time.sleep(2)            # while R does work in second process

    print("Finished after ", time.time() - start_time, " seconds")
