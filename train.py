import os
import glob
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm
from shutil import copyfile

from datetime import datetime

from data_loader import foodSpaceLoader, error_catching_loader, my_collate, collate_fn
from args import get_parser
from models import FoodSpaceNet as FoodSpaceNet
from utils import PadToSquareResize, AverageMeter, worker_init_fn
from loss_func import calculate_loss
from eval import validate

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
opts.gpu = list(map(int, opts.gpu.split(',')))
print('Using GPU(s): ' + ','.join([str(x) for x in opts.gpu]))
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in opts.gpu])
print(torch.cuda.is_available())
# =============================================================================
ranking_loss = torch.nn.MarginRankingLoss(margin=0.3)

class SubsetSequentialSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def main():
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    iter = 0

    # Track results on tensorboard
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print('Saving to: ' + timestamp)
    logdir = "tensorboard/" + timestamp + '__' + os.path.basename(__file__).replace('.py', '') + "/"
    logdir = os.path.join(os.path.dirname(__file__), logdir)
    modeldir = logdir + "models/"
    opts.snapshots = modeldir
    if not os.path.isdir("tensorboard/"): os.mkdir("tensorboard/")
    if not os.path.isdir(logdir): os.mkdir(logdir)
    if not os.path.isdir(modeldir): os.mkdir(modeldir)
    copyfile('models.py', os.path.join(logdir,'models.py'))
    train_writer = SummaryWriter(logdir + '/train')
    train_writer_text = open(os.path.join(logdir, 'train.txt'), 'a')
    test_writer_text = open(os.path.join(logdir, 'test.txt'), 'a')

    with open(os.path.join(logdir, 'opts.txt'), 'w') as file:
        for arg in vars(opts):
            file.write(arg + ': ' + str(getattr(opts, arg)) + '\n')
    model = FoodSpaceNet(opts)
    model.cuda()


    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # preparing the training loader
    train_data = foodSpaceLoader(opts.img_path,
                                  transforms.Compose([
                                      transforms.RandomChoice([
                                          PadToSquareResize(resize=256, padding_mode='random'),
                                          transforms.Resize((256, 256))]),
                                      transforms.RandomRotation(10),
                                      transforms.RandomCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalize]),
                                  data_path=opts.data_path,
                                  partition='train',
                                  maxInsts=opts.maxInsts)
    print('Training loader prepared.')

    # preparing the valitadion loader
    val_data = foodSpaceLoader(opts.img_path,
                            transforms.Compose([
                                PadToSquareResize(resize=256, padding_mode='reflect'),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize]),
                            data_path=opts.data_path,
                            partition='val',
                            loader=error_catching_loader)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=opts.batch_size, shuffle=False,
        # sampler=SubsetSequentialSampler(np.arange(1000)),
        num_workers=opts.workers, pin_memory=True)
    print('Validation loader prepared.')

    # creating different parameter groups
    # model.instNet_.doc_encoder.atten_layer_sent.u
    # model.instNet_.doc_encoder.sent_encoder.atten_layer.u
    # model.attentionVisual.w_context_vector
    # vision_params = list(map(id, model.visionMLP.parameters()))
    vision_params = [kv[1] for kv in model.named_parameters() if kv[0].split('.')[0] in ['visionMLP']]
    attention_params = [kv[1] for kv in model.named_parameters() if 'atten' in kv[0]]
    # embs_params = list(map(id, model.embs.parameters()))
    # base_params   = filter(lambda p: id(p) not in vision_params+embs_params, model.parameters())
    base_params = [kv[1] for kv in model.named_parameters() if (kv[0].split('.')[0] not in ['visionMLP','embs'] and 'atten' not in kv[0])]


    optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': vision_params, 'lr': opts.lr*opts.freeVision },
                {'params': model.embs.parameters(), 'lr': opts.lr*opts.freeEmbs },
                {'params': attention_params, 'lr': 0 }
            ], lr=opts.lr*opts.freeRecipe)
    if len(opts.gpu) > 1:
        model = torch.nn.DataParallel(model)

    if opts.resume:
        if os.path.isfile(opts.resume):
            print("=> loading checkpoint '{}'".format(opts.resume))
            checkpoint = torch.load(opts.resume)
            opts.start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            best_rec = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer = checkpoint['optimizer']
            iter = len(train_data)/opts.batch_size * opts.start_epoch
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opts.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opts.resume))
            best_val = float('inf')
            best_rec = 0
    else:
        best_val = float('inf')
        best_rec = 0
    
    since_improvement = 0

    print('There are %d parameter groups' % len(optimizer.param_groups))
    print('\tInitial base params lr: %f' % optimizer.param_groups[0]['lr'])
    print('\tInitial vision params lr: %f' % optimizer.param_groups[1]['lr'])
    print('\tInitial word2vec params lr: %f' % optimizer.param_groups[2]['lr'])

    cudnn.benchmark = True
    warmup_lr = np.linspace(opts.lr * 0.01, opts.lr, opts.warmup)

    for epoch in range(opts.start_epoch, opts.epochs):
        print('Started epoch {}/{}'.format(epoch+1, opts.epochs))
        inds = np.arange(len(train_data)).tolist()
        random.shuffle(inds)
        inds = np.asanyarray(inds)
        print('\tStarted training...')

        # preparing the training loader
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=opts.batch_size,
                                                   shuffle=False,
                                                   sampler=SubsetSequentialSampler(inds),
                                                   num_workers=opts.workers,
                                                   pin_memory=True,
                                                   worker_init_fn=worker_init_fn,
                                                   collate_fn=my_collate)

        if epoch < opts.warmup and opts.warmup >= 2:
            opts.lr = warmup_lr[epoch]
            optimizer.param_groups[0]['lr'] = opts.lr * opts.freeRecipe
            optimizer.param_groups[1]['lr'] = opts.lr * opts.freeVision
            optimizer.param_groups[2]['lr'] = opts.lr * opts.freeEmbs

        iter = train(train_loader, train_data, model,  optimizer, epoch, iter, train_writer, train_writer_text)

        if str(epoch + 1) in opts.patience.split(','):
            state = {'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'best_val': best_val,
                     'val_medR': 0,
                     'optimizer': optimizer,
                     'since_improvement': since_improvement,
                     'freeVision': opts.freeVision,
                     'curr_val': 0,
                     'curr_recall': 0}
            filename = opts.snapshots + 'model_preFreeWeigths_epoch{}.pth.tar'.format(state['epoch'])
            torch.save(state, filename)

            opts.freeVision = True
            opts.freeRecipe = True
            opts.freeEmbs = True
            optimizer.param_groups[0]['lr'] = opts.lr * opts.freeRecipe
            optimizer.param_groups[1]['lr'] = opts.lr * opts.freeVision
            optimizer.param_groups[2]['lr'] = opts.lr * opts.freeEmbs
            optimizer.param_groups[3]['lr'] = opts.lr
        if str(epoch + 1) in opts.patienceLR.split(','):
            state = {'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'best_val': best_val,
                     'val_medR': 0,
                     'optimizer': optimizer,
                     'since_improvement': since_improvement,
                     'freeVision': opts.freeVision,
                     'curr_val': 0,
                     'curr_recall': 0}
            filename = opts.snapshots + 'model_preLR_{}-{}_epoch{}.pth.tar'.format(opts.lr,opts.lr*0.1, state['epoch'])
            torch.save(state, filename)
            opts.lr = opts.lr * 0.1
            optimizer.param_groups[0]['lr'] = opts.lr * opts.freeRecipe
            optimizer.param_groups[1]['lr'] = opts.lr * opts.freeVision
            optimizer.param_groups[2]['lr'] = opts.lr * opts.freeEmbs
            optimizer.param_groups[3]['lr'] = opts.lr

        if (epoch+1) % opts.valfreq == 0:
            val_medR, val_recall, val_meanR, val_meanDCG = validate(val_loader, model, opts)

            if val_medR > best_val:
                since_improvement += 1
            else:
                since_improvement = 0

            cum_recall = sum([v for k, v in val_recall.items()])
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
                'val_medR': val_medR,
                'optimizer': optimizer,
                'since_improvement': since_improvement,
                'freeVision': opts.freeVision,
                'curr_val': val_medR,
                'curr_recall': val_recall
                }

            filename = opts.snapshots + 'model.pth.tar'
            torch.save(state, filename)
            if val_medR < best_val:
                if glob.glob(opts.snapshots + 'model_BEST_VAL*'): os.remove(glob.glob(opts.snapshots + 'model_BEST_VAL*')[0])
                filename = opts.snapshots + 'model_BEST_VAL_e%03d_v-%.3f_cr-%.4f.pth.tar' % (
                state['epoch'], state['val_medR'], cum_recall)
                torch.save(state, filename)
            if cum_recall > best_rec:
                if glob.glob(opts.snapshots + 'model_BEST_REC*'): os.remove(glob.glob(opts.snapshots + 'model_BEST_REC*')[0])
                filename = opts.snapshots + 'model_BEST_REC_e%03d_v-%.3f_cr-%.4f.pth.tar' % (
                state['epoch'], state['val_medR'], cum_recall)
                torch.save(state, filename)

            best_val = min(val_medR, best_val)
            best_rec = max(cum_recall, best_rec)

            train_writer.add_scalar("medR", val_medR, epoch)
            train_writer.add_scalar("meanR", val_meanR, epoch)
            train_writer.add_scalar("meanDCGR", val_meanDCG, epoch)
            train_writer.add_scalar("recall@1", val_recall[1], epoch)
            train_writer.add_scalar("recall@5", val_recall[5], epoch)
            train_writer.add_scalar("recall@10", val_recall[10], epoch)
            train_writer.add_scalar("recall@1-5-10", val_recall[10]+val_recall[5]+val_recall[1], epoch)
            train_writer.flush()

            test_writer_text.write(str(iter)+','+str(val_medR)+'\n')
            test_writer_text.flush()
            print('\t** Validation: %f (best) - %d (since_improvement)' % (best_val, since_improvement))


def train(train_loader, train_data, model, optimizer, epoch, iter, train_writer, train_writer_text):
    losses = AverageMeter()
    model.train()
    for i, (input, rec_ids) in enumerate(tqdm(train_loader, total=len(train_loader))):
        iter += 1

        output = model(input, opts)

        loss = calculate_loss(output, opts)

        with torch.no_grad():
            losses.update(loss.item(), input[0].size(0))
            if i % 100 == 0:
                train_writer.add_scalar("cos_loss", loss.item(), iter)
                train_writer.flush()
                train_writer_text.write(str(iter)+','+str(loss.item())+'\n')
                train_writer_text.flush()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i==10:
        #     break
        del loss

    print('\t\tSubepoch: {0}\t'
              'cos_loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'vision ({visionLR}) - recipe ({recipeLR}) - word2vec ({embsLR}) - attention ({attLR})\t'.format(
               epoch+1, loss=losses,
               visionLR=optimizer.param_groups[1]['lr'],
               recipeLR=optimizer.param_groups[0]['lr'],
               embsLR=optimizer.param_groups[2]['lr'],
                    attLR=optimizer.param_groups[3]['lr']))
    return iter


def save_checkpoint(state):
    filename = opts.snapshots + 'model.pth.tar'
    torch.save(state, filename)


if __name__ == '__main__':
    main()


