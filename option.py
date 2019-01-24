import argparse

def get_args():
    parser = argparse.ArgumentParser(description='OPINE-Net')
    description = 'OPINE-Net pytorch implementation'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--start_epoch', type=int, default=0,
                        help='')
    parser.add_argument('-e', '--end_epoch', type=int, default=200,
                        help='')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                        help='')
    parser.add_argument('--layer_num', type=int, default=9,
                        help='')
    parser.add_argument('--group_num', type=int, default=1,
                        help='')
    parser.add_argument('--phi_weight', type=float, default=0.1,
                        help='')
    parser.add_argument('--cs_ratio', type=int, default=25,
                        help='')
    parser.add_argument('-gpu','--gpu_list', type=str, default='0',
                        help='')
    parser.add_argument('-b','--batch_size', type=int, default=64,
                        help='')
    parser.add_argument('-n','--nrtrain', type=int, default=88912,
                        help='')
    parser.add_argument('-input','--input_data', type=str, default='Training_Data_0_20_Img91.mat',
                        help='')
    args = parser.parse_args()
    return args