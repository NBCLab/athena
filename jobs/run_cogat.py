# -*- coding: utf-8 -*-
"""
@author: salo
"""

import sys
sys.path.append('/home/data/nbc/athena/athena/')
import data_preparation as dp

data_dir = '/home/data/nbc/athena/athena-data2/'
dp.generate_gazetteer(data_dir)
