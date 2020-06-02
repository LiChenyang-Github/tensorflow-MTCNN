
# coding: utf-8

# In[5]:


from model import P_Net,R_Net,O_Net, P_Net_9, P_Net_8, \
    R_Net_conv3, O_Net_conv3, P_Net_v1, R_Net_v1, O_Net_v1, R_Net_fcn, R_Net_fcn_v1, O_Net_fcn_v1, \
    P_Net_aspect_24_12, P_Net_aspect_18_12

R_Net_aspect_24_12 = R_Net
O_Net_aspect_24_12 = O_Net

import argparse
import os
import sys
import config as FLAGS
from train_model import train, train_multi_tfrecords
net_factorys=[P_Net,R_Net,O_Net]

import pdb
import time
import shutil
import os.path as osp

# In[ ]:


def main(args):
    size=args.input_size
    aspect=args.aspect

    ###
    if not args.use_multi_tfrecords:
        if args.suffix is None:
            if size == 24 and args.net_name is not None:
                base_dir=os.path.join('../data_mine/',"{}_{}".format(str(size), \
                    args.net_name.replace('R_Net', 'P_Net')))
            elif size == 48 and args.net_name is not None:
                base_dir=os.path.join('../data_mine/',"{}_{}".format(str(size), \
                    args.net_name.replace('O_Net', 'R_Net')))
            else:
                base_dir=os.path.join('../data_mine/',str(size))
        else:
            if size == 24 and args.net_name is not None:
                base_dir=os.path.join('../data_mine/','{}_{}_{}'.format(str(size), \
                    args.suffix, args.net_name.replace('R_Net', 'P_Net')))
            elif size == 48 and args.net_name is not None:
                base_dir=os.path.join('../data_mine/','{}_{}_{}'.format(str(size), \
                    args.suffix, args.net_name.replace('O_Net', 'R_Net')))
            else:
                base_dir=os.path.join('../data_mine/','{}_{}'.format(str(size), args.suffix))
    else:
        assert isinstance(FLAGS.suffix_list, list) and \
                len(FLAGS.suffix_list) > 1 and \
                len(FLAGS.suffix_list) == len(FLAGS.batch_ratio)
        if size in [12, 9, 8]:
            base_dir = [osp.join('../data_mine/','{}_{}'.format(str(size), x)) for x in FLAGS.suffix_list]
        elif size in [24, 48]:
            if size == 24 and args.net_name is not None:
                base_dir = [
                    osp.join('../data_mine/', '{}_{}_{}'.format(str(size), \
                    '-'.join(FLAGS.suffix_list), args.net_name.replace('R_Net', 'P_Net')), x) \
                    for x in FLAGS.suffix_list
                    ]
            elif size == 48 and args.net_name is not None:
                base_dir = [
                    osp.join('../data_mine/', '{}_{}_{}'.format(str(size), \
                    '-'.join(FLAGS.suffix_list), args.net_name.replace('O_Net', 'R_Net')), x) \
                    for x in FLAGS.suffix_list
                    ]
            else:
                base_dir = [osp.join('../data_mine/', '{}_{}'.format(str(size), '-'.join(FLAGS.suffix_list)), x) \
                    for x in FLAGS.suffix_list]

    if args.net_name is None:
        if size==12:
            net='PNet'
            net_factory=net_factorys[0]
            end_epoch=FLAGS.end_epoch[0]
        elif size==9:
            net='PNet_9'
            net_factory=P_Net_9
            end_epoch=FLAGS.end_epoch[0]
        elif size==8:
            net='PNet_8'
            net_factory=P_Net_8
            end_epoch=FLAGS.end_epoch[0]
        elif size==24:
            net='RNet'
            net_factory=net_factorys[1]
            end_epoch=FLAGS.end_epoch[1]
        elif size==48:
            net='ONet'
            net_factory=net_factorys[2]
            end_epoch=FLAGS.end_epoch[2]
    else:
        net = args.net_name
        net_factory = eval(net)
        if size in [12, 9, 8]:
            end_epoch=FLAGS.end_epoch[0]
        elif size == 24:
            end_epoch=FLAGS.end_epoch[1]
        elif size == 48:
            end_epoch=FLAGS.end_epoch[2]
        else: 
            raise

    ###
    if not args.use_multi_tfrecords:
        if args.suffix is None:
            model_path=os.path.join('../model_mine/',net)
        else:
            model_path=os.path.join('../model_mine/', args.suffix, net)
    else:
        model_path=os.path.join('../model_mine/', '-'.join(FLAGS.suffix_list), net)

    # pdb.set_trace()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    config_backup_dir = osp.join(model_path, "config_{}.py".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
    shutil.copy(FLAGS.__file__, config_backup_dir)

    # pdb.set_trace()


    prefix=os.path.join(model_path,net)
    display=FLAGS.display
    lr=FLAGS.lr
    # train(net_factory,prefix,end_epoch,base_dir,display,lr)
    if not args.use_multi_tfrecords:
        train(net_factory,prefix,end_epoch,base_dir,display,lr,args.suffix,pretrained=FLAGS.pretrained,\
            resume=FLAGS.resume,size=size,net=net,exclude_vars=FLAGS.exclude_vars,aspect=aspect)
    else:
        train_multi_tfrecords(net_factory,prefix,end_epoch,base_dir,display,lr,\
            FLAGS.suffix_list,FLAGS.batch_ratio,pretrained=FLAGS.pretrained, \
            resume=FLAGS.resume,size=size,net=net,exclude_vars=FLAGS.exclude_vars)



# In[ ]:
        

def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('input_size', type=int,
                        help='The input size for specific net')

    parser.add_argument('--suffix', type=str, default=None,
                        help='The suffix for the folder')

    parser.add_argument('--use_multi_tfrecords', type=bool, default=False,
                        help='Whether to use multi tfrecords')

    parser.add_argument('--net_name', type=str, default=None,
                        help='The name for the net.')

    parser.add_argument('--aspect', nargs='+', type=int, default=None,
                        help='Specify the (height, width) when the input is not square.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

