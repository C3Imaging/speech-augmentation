import os
import sys
import logging


def setup_logging(arg_folder, filename, console=False):
    """Set up logging to a logfile and optionally to the console also, if console param is True."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # track INFO logging events (default was WARNING)
    root_logger.handlers = []  # clear handlers
    root_logger.addHandler(logging.FileHandler(os.path.join(arg_folder, filename), 'w+'))  # handler to log to file
    root_logger.handlers[0].setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))  # log level and message

    if console:
        root_logger.addHandler(logging.StreamHandler(sys.stdout))  # handler to log to console
        root_logger.handlers[1].setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
 

def BinarySearch(L, x) -> bool:
    """Find leftmost occurrence of value x in sorted list L using binary search and return True if found"""
    first = 0
    last = len(L)-1
    found = False

    while first<=last and not found:
         midpoint = (first + last)//2
         if L[midpoint] == x:
             found = True
         else:
             if x < L[midpoint]:
                 last = midpoint-1
             else:
                 first = midpoint+1

    return found
