import time


def timeit_local_obj(foo):
    def wrapper(*args,  **kwargs):
        start = time.time()
        res = foo(*args, **kwargs),
        args[0].time_local_obj += time.time() - start
        return res[0]
    return wrapper


def timeit(foo):
    def wrapper(*args,  **kwargs):
        start = time.time()
        return foo(*args, **kwargs), time.time() - start
    return wrapper


def timeit_local_retraction(foo):
    def wrapper(*args,  **kwargs):
        start = time.time()
        res = foo(*args, **kwargs),
        args[0].time_local_ret += time.time() - start
        return res[0]
    return wrapper


def timeit_local_projection(foo):
    def wrapper(*args,  **kwargs):
        start = time.time()
        res = foo(*args, **kwargs),
        args[0].time_local_proj += time.time() - start
        return res[0]
    return wrapper