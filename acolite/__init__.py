from acolite import landsat
from acolite import sentinel2
from acolite import sentinel3
from acolite import planet
from acolite import pleiades
from acolite import worldview
from acolite import venus

from acolite import chris
from acolite import prisma
from acolite import hico
from acolite import hyperion
from acolite import desis

from acolite import gf

from acolite import ac
from acolite import aerlut
from acolite import output
from acolite import shared
from acolite import dem

from acolite import tact
from acolite import acolite
from acolite import adjacency

from acolite import gem
from acolite import parameters

import os
code_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(code_path)

## find config file
if not os.path.exists('{}{}config'.format(path, os.path.sep)):
    ## check if binary distribution
    if '{}dist{}acolite'.format(os.path.sep,os.path.sep) in path:
        ## two levels for this file
        for i in range(2): path = os.path.split(path)[0]

cfile='{}{}config{}config.txt'.format(path,os.path.sep,os.path.sep)
config = shared.import_config(cfile)
config['code_path'] = code_path
config['path'] = path

## update version info
if 'version' in config:
    version = 'Generic Version {}'.format(config['version'])
else:
    version = 'Generic GitHub Clone'

## test whether we can find the relative paths
for t in config:
    if t in ['EARTHDATA_u', 'EARTHDATA_p']: continue
    if (os.path.exists(config[t])):
        continue
    tmp = path + os.path.sep + config[t]
    config[t] = os.path.abspath(tmp)

## read parameter scaling and settings
param = {'scaling':acolite.parameter_scaling()}
import json
with open(config['parameter_cf_attributes'], 'r', encoding='utf-8') as f:
    param['attributes'] = json.load(f)

## set up earthdata login
if ('EARTHDATA_u' in config) & ('EARTHDATA_p' in config):
    if len(config['EARTHDATA_u']) > 0: os.environ['EARTHDATA_u'] = config['EARTHDATA_u']
    if len(config['EARTHDATA_p']) > 0: os.environ['EARTHDATA_p'] = config['EARTHDATA_p']
