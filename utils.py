import torch.nn.init as init
import numpy as np

def print_paras_info(model):
    num_count = 0
    para_num = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())
        para_num = para_num + np.prod(para.size())

    print('\nlayer num is %d, para num is %d\n' % (num_count, para_num))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight)
