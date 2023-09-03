"""
Training a CLIP model
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from datetime import timedelta
import torchvision
from torchvision import models
from torch.nn.parallel import DataParallel
import utils
import logging
import clip

import pdb

def parse_args(argv):
    """
    Parsing input arguments.
    """

    # Arguments
    parser = argparse.ArgumentParser(description='Training on RSI datasets.')

    parser.add_argument('--work-dir', type=str, default='./work_dir/', required=False,
                        help='Model directory to use to save models (default=%(default)s)')
        
    # miscellaneous args
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')

    # data args
    parser.add_argument('--dataset', type=str, default='UCMerced_LandUse',
                        help='Dataset (default=%(default)s)')
    parser.add_argument('--season', type=str, default=None, required=False, 
                        help='only used for sen12ms dataset')
    parser.add_argument('--n-shots', default=10, type=int, required=False,
                        help='Number of classes to train with')
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for the dataloader (default=%(default)s)')
    parser.add_argument('--pseudo-version', default='1', type=str, required=False,
                        help='Number ofs classes to train with')
    
    # model args
    parser.add_argument('--vis-model', type=str, required=False, default='ViT-L/14',
                        help='Vision Model')
    parser.add_argument('--gpu', type=str, required=False, default='0',
                        help='GPU assigned')
    
    # training args
    parser.add_argument('--nbatches', default=300, type=int, required=False,
                        help='Maximum number of batches to see per session (default=%(default)s)')
    parser.add_argument('--batch-size', default=24, type=int, required=False,
                        help='Batch size to use (default=%(default)s)')
    parser.add_argument('--decay-step', default=20, type=int, required=False,
                        help='decay-step to use (default=%(default)s)')
    parser.add_argument('--decay', default=0.7, type=float, required=False,
                        help='decay rate to use (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0005, type=float, required=False,
                        help='Weight decay (default=%(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, required=False,
                        help='Momentum (default=%(default)s)')
    parser.add_argument('--patience', type=int, default=3, required=False,
                        help='Use patience while training (default=%(default)s)')
    parser.add_argument('--load-from', type=str, default='',
                        help='Resume training with the last saved model (default=%(default)s)')


    args = parser.parse_args(argv)

    return args

# prompt = "This is a photo of a {}."
# prompt = "This is a satellite image of a {}."
# prompt = "This is a land use image of a {}."
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

    with torch.no_grad():
        cnt = 0
        for data in test_loader:
            images, targets, _ = data
            images = images.to(device)
            targets = targets.to(device).squeeze()
            
            image_features = model.encode_image(images)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            top5_hits, top1_hits = utils.calculate_metrics(similarity, targets)
            total_top1_hits += top1_hits
            total_top5_hits += top5_hits
            N += targets.shape[0]

    top1_avg = 100 * (float(total_top1_hits) / N)
    top5_avg = 100 * (float(total_top5_hits) / N)

    return top1_avg, top5_avg

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 
        
def run_training(args):
    """
    Main training routine.
    """
    def log_string(str):
        logger.info(str)
        print(str)
        
    # Fixing random seed
    utils.seed_everything(args.seed)
    
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

    # Checking directory folder
    if not os.path.exists(args.work_dir):
        print("Output model directory [%s] not found. Creating..." % args.work_dir)
        os.makedirs(args.work_dir)
    os.system('cp train_zero.py {}'.format(args.work_dir))
    
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (args.work_dir, 'train'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    # Setting up dataset
    log_string("Loading data : [%s]" % args.dataset)
    
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    model, preprocess = clip.load(args.vis_model,device=device,jit=False) #Must set jit=False for training

    # Optionally resume training
    if os.path.isfile(args.load_from):
        log_string("Loading pretrained model : [%s]" % args.load_from)
        checkpoint = torch.load(open(args.load_from, 'rb'), map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
    if device == torch.device("cpu"):
        model.float()
    else :
        clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    model.to(device)
    # model = DataParallel(model)

    # Defining Loss and Optimizer
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_step, gamma=args.decay)
    
    # Instantiating data loaders
    # train_transforms = T.Compose([
    #     T.RandomHorizontalFlip(),
    #     T.CenterCrop(224),
    #     T.ToTensor(),
    #     T.Normalize((0.48422758, 0.49005175, 0.45050276), (0.17348297, 0.16352356, 0.15547496)), #recompute
    # ])
    
    train_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        # T.ColorJitter(0.05, 0.05, 0.05),
        # T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    
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
        from data.sen12ms_dataset import SEN12MS, ToTensor, Normalize, CenterCrop, SEN12_Pseudo_Dataset

        imgTransform = T.Compose([
            ToTensor(),
            Normalize(bands_mean, bands_std),
            CenterCrop(224),
        ])

        # test multi_label part with normalization
        print("\n\nSEN12MS train")
        
        if args.season is None:
            season = 'all_seasons'
        else:
            season = args.season
            assert season in ['sprint', 'summer', 'fall', 'winter']

        train_dataset = SEN12_Pseudo_Dataset(data_root, transform=imgTransform, 
                     txt_path='pseudo_v{}_train_{}shot.txt'.format(args.pseudo_version, args.n_shots))
        
        test_dataset = SEN12MS(data_root, list_dir, imgTransform, season=season,
                     label_type="single_label", threshold=0.1, subset="test", 
                     use_s1=False, use_s2=False, use_RGB=True, IGBP_s=True,
                     txt_path='test_{}.txt'.format(season))
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
        train_dataset = RSDataset(data_root=data_root, txt_path='pseudo_v{}_train_{}shot.txt'.format(args.pseudo_version, args.n_shots), transform=train_transforms)
        test_dataset = RSDataset(data_root=data_root, txt_path='all.txt', transform=test_transforms)
    
    log_string('train size: {}/test size: {}'.format(len(train_dataset), len(test_dataset)))
    
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=24, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)


    # Training
    starting_time = time.time()

    # Main loop
    tstart = time.time()

    log_string("Starting training...")
    log_string("Number of batches: [%d]" % (len(train_loader.dataset)/args.batch_size))

    n_batch = 0
    loss = None

    while n_batch <= args.nbatches:
        # Training the model
        total_top1_hits, total_top5_hits, N = 0, 0, 0
        top1_avg, top5_avg = 0, 0
        
        for images, targets, _ in train_loader:
            
            ## Making a checkpoint
            if n_batch % 100 == 0:

                # Measuring model test-accuracy
                top1_test_acc, top5_test_acc = evaluate(model, test_loader, device)
                log_string('Evaluattion {:03}/{:03}, top1_test_acc: {:.3f}, top5_test_acc: {:.3f}'.format(n_batch, args.nbatches, top1_test_acc, top5_test_acc))
                
                if n_batch == 300:
                    saved_model = os.path.join(args.work_dir, "batch_%d.ckpt" % (n_batch))
                    log_string("Saved model at: [" + saved_model + "]")
                    state = {
                        "top1_train_acc": top1_avg,
                        "top5_train_acc": top5_avg,
                        "top1_test_acc": top1_test_acc,
                        "top5_test_acc": top5_test_acc,
                        "state_dict": model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, saved_model)
                
            # pdb.set_trace()
            model.train()
            optimizer.zero_grad()
            
            images, targets = images.to(device), targets.to(device)
            texts = [prompt + test_loader.dataset.classes[t] for t in targets]
            texts = torch.stack([clip.tokenize(t) for t in texts])
            texts = texts.squeeze().to(device)
            
            ## Forward + backward + optimize
            logits_per_image, logits_per_text = model(images, texts) #logits_per_image_orig
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            
            loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            loss.backward()
            
            if device == torch.device("cpu"):
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            ## Logging results
            outputs = logits_per_image.softmax(dim=-1)
            top5_hits, top1_hits = utils.calculate_metrics(outputs, ground_truth)
            total_top1_hits += top1_hits
            total_top5_hits += top5_hits
            N += images.shape[0]
            top1_avg = 100 * (float(total_top1_hits) / N)
            top5_avg = 100 * (float(total_top5_hits) / N)
            
            log_string('Training {:03}/{:03} | loss = {:.4f} | top-1 acc = {:.3f} | top-5 acc = {:.3f}'.format(n_batch, args.nbatches, loss.item(), top1_avg, top5_avg))

            n_batch += 1
            scheduler.step()

            if (n_batch >= args.nbatches): break

        # Logging results
        current_elapsed_time = time.time() - starting_time
        log_string('{:03}/{:03} | {} | Train : loss = {:.4f} | top-1 acc = {:.3f} | top-5 acc = {:.3f}'.
                format(n_batch, args.nbatches,
                        timedelta(seconds=round(current_elapsed_time)),
                        loss, top1_avg, top5_avg))

    # Final output
    log_string('[Elapsed time = {:.1f} mn]'.format((time.time() - tstart) / 60))
    log_string('Done!')

    log_string('-' * 108)


def main(argv=None):
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
