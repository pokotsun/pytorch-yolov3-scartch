from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    block = {}
    blocks = []
    with open(cfgfile, 'r') as f:
        lines = f.read().split('\n')
        lines = [x.rstrip().lstrip() for x in lines if len(x) > 0 and x[0] != '#']
        
        for line in lines:
            if line[0] == "[":
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block["type"] = line [1:-1].rstrip()
            else: 
                key, value = [ x.strip() for x in line.split("=") ]
                block[key] = value 
        blocks.append(block) # append last section
    return blocks
                
