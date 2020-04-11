import yaml
import os
cur_file = __file__

env= None

if  'env' in os.environ:
    env = os.environ['env']
if env is not None :
    file_name = 'config_{}.yaml'.format(env)
else :
    file_name =  'config.yaml'

with open (os.path.join(os.path.dirname(cur_file), file_name) )as f:
    Config = yaml.load(f)
print(Config)