import time

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 调用原函数
        end_time = time.time()  # 记录结束时间
        print(f"{func.__name__}运行时间: {end_time - start_time:.6f}秒")
        return result
    return wrapper

