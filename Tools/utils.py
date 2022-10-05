import os
import sys
import logging

def setup_logging(arg_folder):
    """Set up logging to both console and a logfile."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # track INFO logging events (default was WARNING)
    root_logger.handlers = []  # clear handlers
    root_logger.addHandler(logging.StreamHandler(sys.stdout))  # handler to log to console
    root_logger.addHandler(logging.FileHandler(os.path.join(arg_folder, 'wav2vec2_alignments_run.log'), 'w+'))  # handler to log to file also
    root_logger.handlers[0].setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))  # log level and message
    root_logger.handlers[1].setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))