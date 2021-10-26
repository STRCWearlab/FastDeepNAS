# Search space config

## BlockQNN config
# N_CHANNELS = 12
# MAX_INDEX = 11
# N_KERNELS_CONV = [64]
# KERNEL_SIZES_CONV = [1, 3, 5, 7, 9, 11]
# KERNEL_SIZES_POOL = [2, 3, 5]
# TYPES = {0: 'Input',
#          1: 'Convolution',
#          2: 'Max Pooling',
#          3: 'Average Pooling',
#          4: 'Identity',
#          5: 'Elementwise Addition',
#          6: 'Concatenation',
#          7: 'Terminal'}
# BATCH_SIZE = 1000

# Streamlined config
N_CHANNELS = 113
MAX_INDEX = 8
N_KERNELS_CONV = [32, 64, 128, 256]
KERNEL_SIZES_CONV = [1, 2, 3, 5, 8]
KERNEL_SIZES_POOL = [2, 3, 5]
TYPES = {0: 'Input',
         1: 'Convolution',
         2: 'Max Pooling',
         3: 'Concatenation',
         4: 'Terminal'}
BATCH_SIZE = 1000

#Configs
#with win_size=16 - 
#N_KERNELS_CONV = [32,64,128,256]
#KERNEL_SIZES_CONV = [1,2,3,5,8]
#KERNEL_SIZES_POOL = [2,3,5]
#with win_size=32 - 
#N_KERNELS_CONV = [32,64,128,256]
#KERNEL_SIZES_CONV = [1,3,6,11,16]
#KERNEL_SIZES_POOL = [2,3,5]
#with win_size=48 - 
#N_KERNELS_CONV = [32,64,128,256]
#KERNEL_SIZES_CONV = [1,5,9,15,22]
#KERNEL_SIZES_POOL = [2,3,5]
#with win_size=64 - 
#N_KERNELS_CONV = [32,64,128,256]
#KERNEL_SIZES_CONV = [1,6,12,21,32]
#KERNEL_SIZES_POOL = [2,3,5]