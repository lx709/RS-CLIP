import numpy as np
import os
import time
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import argparse
from pkg_resources import packaging

print("Torch version:", torch.__version__)

os.sys.path.append('../')

import clip
import utils

def parse_args(argv):
    """
    Parsing input arguments.
    """

    # Arguments
    parser = argparse.ArgumentParser(description='arguments.')

    # data args
    parser.add_argument('--dataset', type=str, default='ucm', required=False, 
                        help='Dataset (default=%(default)s)')
    parser.add_argument('--season', type=str, default=None, required=False, 
                        help='only used for sen12ms dataset')
    parser.add_argument('--topk', default=20, type=int, required=False,
                        help='Number of classes to train with')
    parser.add_argument('--pseudo-version', default='2', type=str, required=False,
                        help='Number ofs classes to train with')
    
    # model args
    parser.add_argument('--vis-model', type=str, required=False, default='ViT-B/32',
                        help='Vision Model')
    parser.add_argument('--gpu', type=str, required=False, default='0',
                        help='GPU assigned')
    
    # training args
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Batch size to use (default=%(default)s)')
    parser.add_argument('--load-from', type=str, default='', required=False,
                        help='Resume training with the last saved model (default=%(default)s)')
    
    args = parser.parse_args(argv)

    return args

prompt = "This is an aerial image of a {}."

def evaluate(model, test_loader, device):
    """
    Evaluating the resulting classifier using a given test set loader.
    """
    model.eval()
    model.float()
    text_descriptions = [prompt.format(label) for label in test_loader.dataset.classes]
    text_tokens = clip.tokenize(text_descriptions).cuda()

    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    total_top1_hits, total_top5_hits, N = 0, 0, 0
    top1_avg, top5_avg = 0, 0
    pre_probs = []
    gts = []
    with torch.no_grad():
        cnt = 0
        for data in test_loader:
            images, targets, _ = data
            images = images.to(device)
            targets = targets.to(device).squeeze()
            
            image_features = model.encode_image(images)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pre_probs.append(similarity.cpu().numpy())
            gts.append(targets.cpu().numpy())
            
            top5_hits, top1_hits = utils.calculate_metrics(similarity, targets)
            total_top1_hits += top1_hits
            total_top5_hits += top5_hits
            N += targets.shape[0]

    top1_avg = 100 * (float(total_top1_hits) / N)
    top5_avg = 100 * (float(total_top5_hits) / N)
    
    pre_probs = np.concatenate(pre_probs, 0)
    return top1_avg, top5_avg, pre_probs, gts

def run_gen(args):
    
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    
    model, preprocess = clip.load(args.vis_model)
    model.cuda().eval()
    
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    test_transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    
    if 'sen12ms' in args.dataset.lower():
        
        
        bands_mean = {'s1_mean': [-11.76858, -18.294598],
                  's2_mean': [1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058,
                              2211.1584, 2154.9846, 2409.1128, 2001.8622, 1356.0801]}

        bands_std = {'s1_std': [4.525339, 4.3586307],
                     's2_std': [741.6254, 740.883, 960.1045, 946.76056, 985.52747,
                                1082.4341, 1057.7628, 1136.1942, 1132.7898, 991.48016]} 

        # data path
        data_root = './data/SEN12MS/'
        list_dir = "./data/SEN12MS/SEN12MS/splits/"    # split lists/ label dirs

        # define image transform
        from data.sen12ms_dataset import SEN12MS, ToTensor, Normalize, CenterCrop

        imgTransform = T.Compose([
            ToTensor(),
            Normalize(bands_mean, bands_std),
            CenterCrop(224),
        ])

        # test multi_label part with normalization
        
        if args.season is None:
            season = 'all_seasons'
        else:
            season = args.season
            assert season in ['sprint', 'summer', 'fall', 'winter']
        
        print("\n\nSEN12MS test")
        test_dataset = SEN12MS(data_root, list_dir, imgTransform, season=season,
                     label_type="single_label", threshold=0.1, subset="test", 
                     use_s1=False, use_s2=False, use_RGB=True, IGBP_s=True)
        
    else:
        if 'ucm' in args.dataset.lower():
            data_root='./data/UCMerced_LandUse/'
        elif 'whurs' in args.dataset.lower():
            data_root='./data/WHU-RS19/'
        elif 'nwpu' in args.dataset.lower():
            data_root='./data/NWPU-RESISC45/'
        elif 'aid' in args.dataset.lower():
            data_root='./data/AID/'
        elif 'eurosat' in args.dataset.lower():
            data_root = './data/EuroSAT/'
        else:
            print('dataset not defined')
            return

        from data.ucm_dataset import RSDataset

        test_dataset = RSDataset(data_root=data_root, txt_path='all.txt', transform=test_transforms)
    
    print('test size', len(test_dataset))


    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)


    device = torch.device("cuda:0")

    if os.path.isfile(args.load_from):
        checkpoint = torch.load(open(args.load_from, 'rb'), map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('file not exist')

    top1_test_acc, top5_test_acc, pre_probs, gts = evaluate(model, test_loader, device)
    print('top1_test_acc: {:.3f}, top5_test_acc: {:.3f}'.format(top1_test_acc, top5_test_acc))

    topk=args.topk

    fp_train = open(data_root + 'pseudo_v{}_train_{}shot.txt'.format(args.pseudo_version, topk), 'w')
    fp_valid = open(data_root + 'pseudo_v{}_valid_{}shot.txt'.format(args.pseudo_version, topk), 'w')
    correct_all = 0.0

    train_cnt, test_cnt = 0, 0
    classes = test_dataset.classes
    for c, name in enumerate(classes):
        pre_probs_per_class = pre_probs[:, c]
        indices = np.argsort(-pre_probs_per_class)[:topk]
        # print(c, name, pre_probs_per_class[indices])
        n_ims = len(indices)

        correct = 0.0
        for ind in indices:
            im, label, im_path = test_dataset[ind]
            correct += (label==c)
            correct_all += (label==c)
            im = np.array(torch.squeeze(im).permute(1, 2, 0))
            fp_train.writelines(im_path + ', ' + str(c) + ', ' + str(label) + '\n')
            train_cnt += 1
    print('pseudo accuracy: {:.4f}'.format(correct_all/(len(classes)*topk)))
    fp_train.close()
    
    if 'sen12ms' in args.dataset.lower():
        all_list = np.loadtxt(data_root + 'test_{}.txt'.format(season), delimiter=' ', dtype='str')
    else:
        all_list = np.loadtxt(data_root + 'all.txt', delimiter=' ', dtype='str')
    
    all_train = np.loadtxt(data_root + 'pseudo_v{}_train_{}shot.txt'.format(args.pseudo_version, topk), delimiter=' ', dtype='str')
    for item in all_list:
        im_path, label = item
        if im_path not in all_train[:, 0]:
            fp_valid.writelines(im_path + ', ' + str(label) + '\n')
            test_cnt += 1

    fp_valid.close()
    print(train_cnt, test_cnt, train_cnt+test_cnt)

def main(argv=None):
    args = parse_args(argv)
    run_gen(args)


if __name__ == "__main__":
    main()