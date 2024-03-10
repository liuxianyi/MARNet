import numpy as np
import torch
import math
from collections import OrderedDict

def avg_p(output, target):
    sorted, indices = torch.sort(output, dim=1, descending=True)
    tp = 0
    s = 0
    for i in range(target.size(1)):
        idx = indices[0,i]
        if target[0,idx] == 1:
            tp = tp + 1
            pre = tp / (i+1)
            s = s + pre
    if tp == 0:
        AP = 0
    else:
        AP = s/tp
    return AP

def cal_ap(y_pred,y_true):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    ap = torch.zeros(y_pred.size(0)) 
    # compute average precision for each class
    for k in range(y_pred.size(0)):
        # sort scores
        scores = y_pred[k,:].reshape([1,-1])
        targets = y_true[k,:].reshape([1,-1])
        # compute average precision
        ap[k] = avg_p(scores, targets)
    return ap

def cal_one_error(output, target):
    num_class, num_instance = output.size(1), output.size(0)
    one_error = 0
    for i in range(num_instance):
        indicator = 0
        Label = []
        not_Label = []
        temp_tar = target[i, :].reshape(1, num_class)
        for j in range(num_class):  
            if (temp_tar[0, j] == 1):
                Label.append(j)
            else:
                not_Label.append(j)
        temp_out = output[i, :].cpu().numpy()
        maximum = max(temp_out)
        index = np.argmax(temp_out)
        for j in range(num_class):
            if (temp_out[j] == maximum):
                if index in Label:
                    indicator = 1
                    break
        if indicator == 0:
            one_error = one_error + 1

    one_error = one_error / num_instance
    return one_error

def cal_coverage(output, target):
    num_class, num_instance = output.size(1), output.size(0)
    cover = 0
    for i in range(num_instance):
        Label = []
        not_Label = []
        temp_tar = target[i,:].reshape(1,num_class)
        Label_size = sum(sum(temp_tar ==torch.ones([1,num_class])))
        for j in range(num_class):  
            if(temp_tar[0,j]==1):
                Label.append(j)
            else:
                not_Label.append(j)
        temp_out = output[i,:]
        _,inde = torch.sort(temp_out)  
        inde = inde.cpu().numpy().tolist()
        temp_min = num_class
        for m in range(Label_size):
            loc = inde.index(Label[m])
            if (loc<temp_min):
                temp_min = loc
        cover = cover + (num_class-temp_min)

    cover_result = (cover/num_instance)-1
    return cover_result


def cal_RankingLoss(output, target):
    num_class, num_instance = output.size(1), output.size(0)
    rankloss = 0
    for i in range(num_instance):
        Label = []  
        not_Label = []  
        temp_tar = target[i,:].reshape(1,num_class)
        Label_size = sum(sum(temp_tar ==torch.ones([1,num_class])))
        for j in range(num_class):
            if (temp_tar[0, j] == 1):
                Label.append(j)
            else:
                not_Label.append(j)
        temp = 0
        for m in range(Label_size):
            for n in range(num_class-Label_size):   
                if output[i,Label[m]]<=output[i,not_Label[n]]: 
                    temp += 1
        if Label_size==0:
            continue
        else:
            rankloss = rankloss + temp / (Label_size * (num_class-Label_size))

    RankingLoss = rankloss / num_instance
    return RankingLoss

def cal_HammingLoss(output, target):
    pre_output = torch.zeros(output.size(0),output.size(1)).cuda(0)
    for i in range(output.size(0)):
        for j in range(output.size(1)):
            if output[i,j]>=0.5:
                pre_output[i,j]=1
            else:
                pre_output[i,j]=0
    num_class, num_instance = output.size(1), output.size(0)
    miss_sum = 0
    for i in range(num_instance):
        miss_pairs = sum(pre_output[i,:]!=target[i,:].cuda(0))
        miss_sum += miss_pairs
    HammingLoss = miss_sum/(num_class*num_instance)

    return HammingLoss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()): # drop iter
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items(): # drop iter
            tb_logger.log_value(prefix + k, v.val, step=step)

class AveragePrecisionMeter(object):
    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        with torch.no_grad():
            self.scores.resize_(offset + output.size(0), output.size(1))
            self.targets.resize_(offset + target.size(0), target.size(1))
            self.scores.narrow(0, offset, output.size(0)).copy_(output)
            self.targets.narrow(0, offset, target.size(0)).copy_(target)


    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)
            # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:

                continue
            if label == 1:
                pos_count += 1
            total_count  += 1
            if label == 1:
                precision_at_i += pos_count/total_count
        if pos_count == 0:
            precision_at_i = 0
        else:
            precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)


    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)
        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1




