import functools
import time

TIME_UNIT_CONVERSION = {
    "s": 1,
    "min": 60,
    "h": 3600,
    "d": 86400,
}


def timeit(func_=None, unit="s", with_args=False, skip_first_arg=False):
    if unit not in TIME_UNIT_CONVERSION:
        raise ValueError(f"'unit' must be one of {set(TIME_UNIT_CONVERSION.keys())}")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            total_time = (end_time - start_time) / TIME_UNIT_CONVERSION[unit]
            message = func.__name__
            if with_args:
                args_for_formatting = args[1:] if skip_first_arg else args
                args_str = ", ".join(
                    f"'{a}'" if isinstance(a, str) else str(a)
                    for a in args_for_formatting
                )
                kwargs_vals = map(
                    lambda v: f"'{v}'" if isinstance(v, str) else str(v),
                    kwargs.values(),
                )
                kwargs_str = ", ".join(
                    f"{k}={v}" for k, v in zip(kwargs.keys(), kwargs_vals)
                )
                message += f"({args_str}, {kwargs_str})"
            message += f" took {total_time:.4f} {unit}"
            print(message)
            return result

        return wrapper

    # Can use @timeit or @timeit(unit=...)
    return decorator(func_) if callable(func_) else decorator
