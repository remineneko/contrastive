from functools import wraps
from time import perf_counter
from typing import Literal, Callable, Optional, Union

from .logger import logger

def timer(
        time_mode: Optional[Literal['s', 'ms', 'ns']] = 'ms',
        function_name: Optional[Union[str, None]] = None
    ) -> Callable:
    """
    Calculates the amount of time needed to execute a function.
    Usage:
        - You can use it to call with a normal function as parameter.

    ```python
    def foo():
        return 1

    time = timer()(foo)
    time()
    ```

        - or you can use it as a decorator (recommended).

    ```python
    @timer()
    def foo():
        return 1

    foo()
    ```

    You can also specify the unit to measure the time. Supported values:

        - `"s"` for seconds.
        - `"ms"` for miliseconds.
        - `"ns"` for nanoseconds.

    ```python
    @timer(time_mode='ms')
    def foo():
        return 1

    foo()
    ```

    You can also change the name of the function shown in the logger; however, avoid over-using this to make the log harder to understand.
        
    ```python
    @timer(function_name='new_foo')
    def foo():
        return 1

    foo()
    ```
    
    By default, the measured time is in seconds.

    Args:
        time_mode (Literal["s","ns"], Optional): The unit to measure the time in. Defaults to "s".
        function_name (Union[str, None], Optional): The name of the function to override. Defaults to None
        verbose (bool, Optional): Whether to output the time or not.
    """

    # Type-checking.
    
    if time_mode not in ['s', 'ms', 'ns']:
        raise ValueError('time_mode must be either \'s\', \'ms\' or \'ns\'.')

    def inner(func: Callable):      
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__ if not function_name else function_name
            logger.info(f"Running function {func_name}.")
            cur_time = perf_counter()
            result = func(*args, **kwargs)
            new_time = perf_counter()
            time_diff = new_time - cur_time
            if time_mode == 'ms':
                time_diff *= 1000 # 1s = 1000ms
            elif time_mode == 'ns':
                time_diff *= 10**9
            logger.info(f"Function {func_name} took {time_diff}{time_mode} to run.")
            return result
        return wrapper
    return inner