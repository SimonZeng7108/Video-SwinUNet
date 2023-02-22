import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, default='output', help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', default='./configs/swin_tiny_patch4_window7_224_lite.yaml', type=str, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5/test_tcm_7")
config = get_config(args)



def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    bolus_metric_list = 0.0
    pharynx_metric_list = 0.0
    len_bolus_data = len(db_test)
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"][0], sampled_batch["label"][0], sampled_batch['case_name'][0]
        
        bolus_metric, pharynx_metric = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)

        bolus_metric_list += np.array(bolus_metric)
        if bolus_metric[0] == 0:

            len_bolus_data -= 1
        pharynx_metric_list += np.array(pharynx_metric)

        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, bolus_metric[0], bolus_metric[1]))
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, pharynx_metric[0], pharynx_metric[1]))

    bolus_metric_list = bolus_metric_list / len_bolus_data
    
    pharynx_metric_list = pharynx_metric_list / len(db_test)

    bolus_performance = bolus_metric_list[0]
    bolus_mean_hd95 = bolus_metric_list[1]
    pharynx_performance = pharynx_metric_list[0]
    pharynx_mean_hd95 = pharynx_metric_list[1]

    average = (bolus_metric_list+ pharynx_metric_list) / 2

    logging.info('Bolus testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (bolus_performance, bolus_mean_hd95))
    logging.info('Pharynx testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (pharynx_performance, pharynx_mean_hd95))

    print(f"Average: dice: {average[0]:.4f},  hd95: {average[1]:.4f}, asd: {average[2]:.4f}, sensitivity: {average[3]:.4f}, specificity: {average[4]:.4f}, precision: {average[5]:.4f}")
    return "Testing Finished!"


if __name__ == "__main__":


    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    snapshot = os.path.join(args.output_dir, 'epoch_best.pth') #epoch_best_weights epoch_best
    if not os.path.exists(snapshot): snapshot = snapshot.replace('epoch_best', 'epoch_'+ '449')

    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet",msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None


    inference(args, net, test_save_path = True)
