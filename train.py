# coding = utf-8
from tensorboardX import SummaryWriter
import os
import time
import shutil
from easydict import EasyDict
from videoDataset import VideoDataset
from torch.utils.data import DataLoader
import yaml
from model.model import *
from utils import *
import logging
import argparse

def parser():
    par = argparse.ArgumentParser(description="training MARNET args parser.")
    par.add_argument('config', type=str, help='absolute path of config')
    return par.parse_args()

def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    # True select the best convolution algorithm
    torch.backends.cudnn.benchmark = False
    with open(args.config) as f:
        opt = yaml.load(f)
    opt = EasyDict(opt['common'])
    opt.learning_rate = opt.learning_rate  
    writer = SummaryWriter(log_dir=opt.curve_tensorb, flush_secs=5)
    logging.basicConfig(level=logging.INFO,  
                        filename=opt.log_dir,
                        filemode='a',  
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s'
                        )

    ## Load data loaders
    traindataset = VideoDataset(opt.train_label_dir,opt.train_visual_dir,opt.train_audio_dir,opt.train_tra_dir,opt.labelgcn_name,split='train')
    train_loader = DataLoader(traindataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)
    testdataset = VideoDataset(opt.test_label_dir,opt.test_visual_dir,opt.test_audio_dir,opt.test_tra_dir,opt.labelgcn_name,split='test')
    test_loader = DataLoader(testdataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, drop_last=True)

    model = MARNet(opt)
    best_accuracy = 0  

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']
            model.load_state_dict(checkpoint['model'])
            print("=> loaded checkpoint '{}' (epoch {}, best_accuracy {})"
                  .format(opt.resume, start_epoch, best_accuracy))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    for epoch in range(opt.epochs):
        model.scheduler.step()
        writer.add_scalar('learning rate on net', model.optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        # train for one epoch
        train(opt,train_loader,model,epoch,writer)
        # evaluate on validation set
        mAP = validate(opt, test_loader, model, epoch, writer) 
       
        # remember best accuracy and save checkpoint
        is_best = mAP > best_accuracy  
        best_accuracy = max(mAP, best_accuracy) 
        if is_best:
            save_checkpoint({ 
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_accuracy': best_accuracy,
                'opt': opt,
            }, is_best, filename='checkpoint_' + str(epoch) + '.pth.tar', prefix=opt.logger_name + '/')


    print(' *** best={best:.3f}'.format(best=best_accuracy))


def train(opt,train_loader,model,epoch,writer):
    print("start to train")
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    # switch to train mode
    model.train_start()
    since = time.time()
    print("start loading data ...")
    print("learning rate:", model.optimizer.state_dict()['param_groups'][0]['lr'])
    loss_epoch = 0
    map = 0
    map_50 = 0
    for i, train_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - since)  
        # Update the model
        y_hat, loss_batch = model.train_emb(*train_data, opt.batch_size)  
        # y_pred = torch.sigmoid(y_hat)
        loss_epoch = loss_batch + loss_epoch
        # measure elapsed time
        batch_time.update(time.time() - since)  
        map_batch = cal_ap(y_hat, train_data[3]).mean()
        map = map + map_batch
        map_50 = map_50 + map_batch
        # end = time.time()
        print('Train: [{0}/{1}]\t'
              'batch_time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'data_time {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss_batch {loss_batch:.4f}\t'
              'map_batch {map_batch:.4f}\t'
              'learning_rate {lr:.7f}'.format(
            i, len(train_loader), batch_time=batch_time, data_time=data_time, loss_batch=loss_batch,
            map_batch=map_batch,lr=model.optimizer.state_dict()['param_groups'][0]['lr']))

    map = map/len(train_loader)
    print('Epoch: [{0}]\t'
          'Train Loss_epoch {loss:.4f} \t'
          'Loss_avg {loss_avg:.4f} \t'
          'mAP {map:.4f}\t'
          'learning_rate {lr:.7f}'.format(epoch, loss=loss_epoch, loss_avg=loss_epoch / len(train_loader), map=map,
                                          lr=model.optimizer.state_dict()['param_groups'][0]['lr']))
    # # Record logs in tensorboard
    if epoch % opt.log_step == 0:
        logging.info(
            'Train: Epoch: [{0}]\t'
            'Loss_avg {loss_avg:.4f} \t'
            'mAP {map:.4f}\t'
            'learning_rate {lr:.7f}'.format(
                epoch, loss_avg=loss_epoch / len(train_loader),
                map=map,lr=model.optimizer.state_dict()['param_groups'][0]['lr']))
    writer.add_scalar('Train_Loss', loss_epoch / len(train_loader), epoch)
    writer.add_scalar('Train_mAP',map,epoch)



def validate(opt,test_loader,model,epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    val_logger = LogCollector()
    end = time.time()
    running_loss = 0
    map = 0
    coverage = 0
    rank_loss = 0
    hamming_loss = 0
    one_error = 0


    with torch.no_grad():
        print("start validate")
        print("start loading val data...")
        model.eval_start()

        for i, test_data in enumerate(test_loader):
            # measure data loading time
            data_time.update(time.time() - end)  
            # make sure val logger is used
            model.logger = val_logger
            # compute the output and loss
            learned_rep, y_hat, loss = model.test_emb(*test_data, opt.batch_size)
            # learned_rep_list.append(learned_rep)
            # label_list.append(test_data[3].argmax(1))
            running_loss = loss + running_loss
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
   
            map_batch = cal_ap(y_hat, test_data[3]).mean()
            map = map + map_batch
            coverage_batch = cal_coverage(y_hat, test_data[3])
            coverage = coverage + coverage_batch
            rank_loss_batch = cal_RankingLoss(y_hat, test_data[3])
            rank_loss = rank_loss_batch + rank_loss
            hamming_loss_batch = cal_HammingLoss(y_hat, test_data[3])
            hamming_loss = hamming_loss + hamming_loss_batch
            one_error_batch = cal_one_error(y_hat, test_data[3])
            one_error = one_error_batch + one_error
            # Record logs in tensorboard
            print('Test: [{0}/{1}]\t'
                  'batch_time {batch_time.val:.3f} (batch_time.avg:.3f)\t'
                  'data_time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss:.4f}\t'
                  'mAP_batch {map_batch: .4f}'
                  .format(
                i, len(test_loader), batch_time=batch_time, data_time=data_time, loss=loss, map_batch = map_batch))

    map = map/len(test_loader)
    coverage = coverage/len(test_loader)
    rank_loss = rank_loss/len(test_loader)
    hamming_loss = hamming_loss/len(test_loader)
    one_error = one_error/len(test_loader)
    print('Epoch: [{0}]\t'
          'Test Loss_epoch {loss:.4f} \t'
          'Loss_avg {loss_avg:.4f} \t'
          'mAP {map:.4f}'
          'coverage {coverage: .4f}'
          'rank_loss {rank_loss: .4f}'
          'hamming_loss {hamming_loss: .4f}'
          'one_error {one_error: .4f}'
          .format(epoch,loss=running_loss, loss_avg =running_loss/len(test_loader), map=map,
          coverage=coverage, rank_loss=rank_loss, hamming_loss=hamming_loss, one_error=one_error))
    writer.add_scalar('Test_Loss', running_loss / len(test_loader), epoch)
    writer.add_scalar('Test_mAP', map, epoch)
    writer.add_scalar('Test_coverage', coverage, epoch)
    writer.add_scalar('Test_rank_loss', rank_loss, epoch)
    writer.add_scalar('Test_hamming_loss', hamming_loss, epoch)
    writer.add_scalar('Test_one_error', one_error, epoch)
    if epoch % opt.log_step == 0:
        logging.info(
            'Test: Epoch: [{0}]\t'
            'Loss_avg {loss_avg:.4f} \t'
            'mAP {map:.4f}\t'
            'coverage {coverage: .4f}'
            'rank_loss {rank_loss: .4f}'
            'hamming_loss {hamming_loss: .4f}'
            'one_error {one_error: .4f}'
            'learning_rate {lr:.7f}'.format(
                epoch, loss_avg=running_loss / len(test_loader),
                map=map, coverage=coverage, rank_loss=rank_loss, hamming_loss=hamming_loss, one_error=one_error,
                lr=model.optimizer.state_dict()['param_groups'][0]['lr']))

    return map


def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar', prefix = ''):
    torch.save(state,prefix + filename)
    if is_best: 
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt,optimizer,epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 10 epochs"""
    lr = opt.learning_rate * (1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    args = parser()
    main(args)