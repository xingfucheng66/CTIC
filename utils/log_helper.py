import logging
import os
import sys
from datetime import datetime


def logger_init(log_file_name='monitor',
                log_level=logging.DEBUG,
                only_file=False):
    # 指定路径
    os.makedirs(log_file_name, exist_ok=True)

    log_path = os.path.join(log_file_name, str(datetime.now())[:19] + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        logging.basicConfig(filename=log_path,
                            level=log_level,
                            format=formatter,
                            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=log_level,
                            format=formatter,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(log_path),
                                      logging.StreamHandler(sys.stdout)]
                            )

def init_log(args):
    logger_init(log_file_name="log", log_level=logging.DEBUG)
    
    logging.info("### all configs ")
    for key, value in args.__dict__.items():
        logging.info(f"### {key} = {value}")