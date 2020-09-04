from time import time as t
tic_whole = t()
import os
tic = t()
from args import define_args
from pose_net import POSE_NET
toc = t()
print('* Import time:', toc - tic)
FLAGS = define_args()

    
def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    tic = t()
    model = POSE_NET(FLAGS)
    toc = t()
    print('* Initializing time:', toc - tic)
    
    if 'train' in FLAGS.mode:
        print('~~~~~~~~~~~~~~~~')
        print('     TRAIN      ')
        print('~~~~~~~~~~~~~~~~')
        model.train()
    elif FLAGS.mode == 'test':
        print('~~~~~~~~~~~~~~~~')
        print('      TEST      ')
        print('~~~~~~~~~~~~~~~~')
        tic = t()
        model.test()
        toc = t()
        print('* model.test time:', toc - tic)
    else:
        print('~~~~~~~~~~~~~~~~~~~~~~')
        print(' MAKE TEST PICKLE ')
        print('~~~~~~~~~~~~~~~~~~~~~~')
        model.create_pickles()

if __name__=='__main__':
    main()

toc_whole = t()
print('whole time', toc_whole - tic_whole)

