import os
from unicodedata import name
names=os.listdir('./results/')
names_temp=os.listdir('../data/results')
print(names==names_temp)