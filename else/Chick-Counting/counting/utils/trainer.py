import os
import logging
from datetime import datetime
from utils.logger import setlogger


class Trainer(object):
    def __init__(self, args):
        #sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
        sub_dir='ckpt'
        self.save_dir = os.path.join(args.save_dir, sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        setlogger(os.path.join(self.save_dir, 'train.log'))  # set logger
        for k, v in args.__dict__.items():  # save args
            logging.info("{}: {}".format(k, v))
        self.args = args

    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        pass

    def train(self):
        """training one epoch"""
        pass
