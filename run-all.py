import os
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("datasetpath", default=None, type=str)
# parser.add_argument("--prediction", default=False, type=str)


# args = parser.parse_args()
##################random######################
# print('shot1 random')
# os.system('python main-nodl.py rte tf-s-qa-r facebook/opt-30b --num_shots 1 --seed 25')
# print('shot4 random')
# os.system('python main-nodl.py rte tf-s-qa-r facebook/opt-30b --num_shots 4 --seed 25')
# print('######################################')

# print('tf-m')
# print('tf-m shot1')
# os.system('python main-nodl.py rte tf-m-qa-r facebook/opt-30b --num_shots 1 --seed 25')
# print(' tf-m shot4')
# os.system('python main-nodl.py rte tf-m-qa-r facebook/opt-30b --num_shots 4 --seed 25')
# print('######################################')

# print('tf-l random')
# print('tf-l shot1')
# os.system('python main-nodl.py rte tf-l-qa-r facebook/opt-30b --num_shots 1 --seed 25')
# print(' tf-l shot4')
# os.system('python main-nodl.py rte tf-l-qa-r facebook/opt-30b --num_shots 4 --seed 25')
# print('######################################')


# print('tf-xl random')
# print('tf-xl shot1')
# os.system('python main-nodl.py rte tf-xl-qa-r facebook/opt-30b --num_shots 1 --seed 25')
# print(' tf-xl shot4')
# os.system('python main-nodl.py rte tf-xl-qa-r facebook/opt-30b --num_shots 4 --seed 25')
# print('######################################')

######################################################################################
#frequent

print('shot1 f')
os.system('python main-nodl.py rte tf-s-qa-f facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 f')
os.system('python main-nodl.py rte tf-s-qa-f facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('tf-m')
print('tf-m shot1')
os.system('python main-nodl.py rte tf-m-qa-f facebook/opt-30b --num_shots 1 --seed 25')
print(' tf-m shot4')
os.system('python main-nodl.py rte tf-m-qa-f facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('tf-l f')
print('tf-l shot1')
os.system('python main-nodl.py rte tf-l-qa-f facebook/opt-30b --num_shots 1 --seed 25')
print(' tf-l shot4')
os.system('python main-nodl.py rte tf-l-qa-f facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')


print('tf-xl f')
print('tf-xl shot1')
os.system('python main-nodl.py rte tf-xl-qa-f facebook/opt-30b --num_shots 1 --seed 25')
print(' tf-xl shot4')
os.system('python main-nodl.py rte tf-xl-qa-f facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')


###########################
print('inline random')

print('shot0 random')
os.system('python main-nodl.py rte tf-s-qa-inline-r facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-s-qa-inline-r facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-s-qa-inline-r facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('tf-m')
print('shot0 random')
os.system('python main-nodl.py rte tf-m-qa-inline-r facebook/opt-30b --num_shots 0 --seed 25')
print('tf-m shot1')
os.system('python main-nodl.py rte tf-m-qa-inline-r facebook/opt-30b --num_shots 1 --seed 25')
print(' tf-m shot4')
os.system('python main-nodl.py rte tf-m-qa-inline-r facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('tf-l random')
print('shot0 random')
os.system('python main-nodl.py rte tf-l-qa-inline-r facebook/opt-30b --num_shots 0 --seed 25')
print('tf-l shot1')
os.system('python main-nodl.py rte tf-l-qa-inline-r facebook/opt-30b --num_shots 1 --seed 25')
print(' tf-l shot4')
os.system('python main-nodl.py rte tf-l-qa-inline-r facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')


print('tf-xl random')
print('shot0 random')
os.system('python main-nodl.py rte tf-xl-qa-inline-r facebook/opt-30b --num_shots 0 --seed 25')
print('tf-xl shot1')
os.system('python main-nodl.py rte tf-xl-qa-inline-r facebook/opt-30b --num_shots 1 --seed 25')
print(' tf-xl shot4')
os.system('python main-nodl.py rte tf-xl-qa-inline-r facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

###########################
print('inline fre')

print('shot0 f')
os.system('python main-nodl.py rte tf-s-qa-inline-f facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-s-qa-inline-f facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-s-qa-inline-f facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('tf-m')
print('shot0 f')
os.system('python main-nodl.py rte tf-m-qa-inline-f facebook/opt-30b --num_shots 0 --seed 25')
print('tf-m shot1')
os.system('python main-nodl.py rte tf-m-qa-inline-f facebook/opt-30b --num_shots 1 --seed 25')
print(' tf-m shot4')
os.system('python main-nodl.py rte tf-m-qa-inline-f facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('tf-l f')
print('shot0 f')
os.system('python main-nodl.py rte tf-l-qa-inline-f facebook/opt-30b --num_shots 0 --seed 25')
print('tf-l shot1')
os.system('python main-nodl.py rte tf-l-qa-inline-f facebook/opt-30b --num_shots 1 --seed 25')
print(' tf-l shot4')
os.system('python main-nodl.py rte tf-l-qa-inline-f facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')


print('tf-xl f')
print('shot0 f')
os.system('python main-nodl.py rte tf-xl-qa-inline-f facebook/opt-30b --num_shots 0 --seed 25')
print('tf-xl shot1')
os.system('python main-nodl.py rte tf-xl-qa-inline-f facebook/opt-30b --num_shots 1 --seed 25')
print(' tf-xl shot4')
os.system('python main-nodl.py rte tf-xl-qa-inline-f facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')



#########################################################
print('AFTER Human')

print('shot0 ')
os.system('python main-nodl.py rte tf-s-human facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-s-human facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-s-human facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('inline Human')

print('shot0 ')
os.system('python main-nodl.py rte tf-m-human facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-m-human facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-m-human facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('inline Human')

print('shot0 ')
os.system('python main-nodl.py rte tf-l-human facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-l-human facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-l-human facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('inline Human')

print('shot0 ')
os.system('python main-nodl.py rte tf-xl-human facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-xl-human facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-xl-human facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

#########################################################
print('Inline Human')

print('shot0 ')
os.system('python main-nodl.py rte tf-s-human-inline facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-s-human-inline facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-s-human-inline facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('inline Human')

print('shot0 ')
os.system('python main-nodl.py rte tf-m-human-inline facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-m-human-inline facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-m-human-inline facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('inline Human')

print('shot0 ')
os.system('python main-nodl.py rte tf-l-human-inline facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-l-human-inline facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-l-human-inline facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')

print('inline Human')

print('shot0 ')
os.system('python main-nodl.py rte tf-xl-human-inline facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-xl-human-inline facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-xl-human-inline facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')


##########################
#########################################################
print('AFTER normal')

print('shot0 ')
os.system('python main-nodl.py rte tf-s facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-s facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-s facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')



print('shot0 ')
os.system('python main-nodl.py rte tf-m facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-m facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-m facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')



print('shot0 ')
os.system('python main-nodl.py rte tf-l facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-l facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-l facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')


print('shot0 ')
os.system('python main-nodl.py rte tf-xl facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-xl facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-xl facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')


#########################################################
print('Inline normal')

print('shot0 ')
os.system('python main-nodl.py rte tf-s-inline facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-s-inline facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-s-inline facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')



print('shot0 ')
os.system('python main-nodl.py rte tf-m-inline facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-m-inline facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-m-inline facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')



print('shot0 ')
os.system('python main-nodl.py rte tf-l-inline facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-l-inline facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-l-inline facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')


print('shot0 ')
os.system('python main-nodl.py rte tf-xl-inline facebook/opt-30b --num_shots 0 --seed 25')
print('shot1 random')
os.system('python main-nodl.py rte tf-xl-inline facebook/opt-30b --num_shots 1 --seed 25')
print('shot4 random')
os.system('python main-nodl.py rte tf-xl-inline facebook/opt-30b --num_shots 4 --seed 25')
print('######################################')
