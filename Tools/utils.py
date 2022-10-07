import os
import sys
import logging
from bisect import bisect_left


def setup_logging_console_folder(arg_folder):
    """Set up logging to both console and a logfile."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # track INFO logging events (default was WARNING)
    root_logger.handlers = []  # clear handlers
    root_logger.addHandler(logging.StreamHandler(sys.stdout))  # handler to log to console
    root_logger.addHandler(logging.FileHandler(os.path.join(arg_folder, 'wav2vec2_alignments_run.log'), 'w+'))  # handler to log to file also
    root_logger.handlers[0].setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))  # log level and message
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
