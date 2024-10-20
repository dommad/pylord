"""Various utility objects"""

# MIT License

# Copyright (C) 2023 Dominik Madej

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import time
import configparser


def open_config(config_file_path: str):
    
    with open(config_file_path, 'r', encoding='utf-8') as config_file:
        config = configparser.ConfigParser()
        config.read_file(config_file)
    
    return config


def fetch_instance(class_name, attribute_name):
    """general fetches for class attributes by name and possibly initializing them"""
    try:
        return getattr(class_name, attribute_name)

    except AttributeError as exc:
        raise ValueError(f"Unsupported or invalid instance type: {class_name}") from exc


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds.")
        return result
    return wrapper


def log_function_call(func):

    def wrapper(*args, **kwargs):
        # print(f"Calling {func.__name__} with args {args} and kwargs {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result

    return wrapper



def _is_numeric(value):
    if not isinstance(value, str):
        return False
    try:
        float(value)
        return True

    except ValueError:
        return False


def largest_factors(n):
    for i in range(n // 2, 0, -1):
        if n % i == 0:
            return n // i, i


class StrClassNameMeta(type):

    def __str__(cls):
        return cls.__name__


class ParserError(Exception):
    """A custom exception class."""
    def __init__(self, message="An error occurred."):
        self.message = message
        super().__init__(self.message)

