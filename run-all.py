import os
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("datasetpath", default=None, type=str)
# parser.add_argument("--prediction", default=False, type=str)


# args = parser.parse_args()

print('tf-s')
# print('shot0')
# os.system('python main-nodl.py rte tf-s-qa facebook/opt-30b --num_shots 0 --seed 25')
# print('shot1')
# os.system('python main-nodl.py rte tf-s-qa facebook/opt-30b --num_shots 1 --seed 25')
print('shot4')
os.system('python main-nodl.py rte tf-s-qa facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('tf-m')
print('shot0')
os.system('python main-nodl.py rte tf-m-qa facebook/opt-30b --num_shots 0 --seed 25')
print('tf-m shot1')
os.system('python main-nodl.py rte tf-m-qa facebook/opt-30b --num_shots 1 --seed 25')
print(' tf-m shot4')
os.system('python main-nodl.py rte tf-m-qa facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('tf-l')
print('shot0')
os.system('python main-nodl.py rte tf-l-qa facebook/opt-30b --num_shots 0 --seed 25')
print('tf-l shot1')
os.system('python main-nodl.py rte tf-l-qa facebook/opt-30b --num_shots 1 --seed 25')
print(' tf-l shot4')
os.system('python main-nodl.py rte tf-l-qa facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')


print('tf-xl')
print('shot0')
os.system('python main-nodl.py rte tf-xl-qa facebook/opt-30b --num_shots 0 --seed 25')
print('tf-xl shot1')
os.system('python main-nodl.py rte tf-xl-qa facebook/opt-30b --num_shots 1 --seed 25')
print(' tf-xl shot4')
os.system('python main-nodl.py rte tf-xl-qa facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')
