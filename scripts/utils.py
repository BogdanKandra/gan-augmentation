import logging
from math import sqrt
import sys
from scripts import config


def get_logger(name: str) -> logging.Logger:
    """ Creates and returns a Logger object for the requested script """
    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)

    return logger

def is_perfect_square(number: int) -> bool:
    """ Checks whether the given number is a perfect square """
    if type(number) is not int:
        raise TypeError('Unexpected type for parameter "number" (Expected <int>, given <{}>)'.format(type(number)))

    return int(sqrt(number)) ** 2 == number
