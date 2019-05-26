#%%

# When ran in jupyter context will allow resolution
# of our modules
import sys
sys.path.append('./src')

from dataset import factory

factory.create()