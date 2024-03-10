# coding=utf-8
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
class VideoDataset(Dataset):
    def __init__(self,label_dir,visual_dir,audio_dir,tra_dir,labelgcn_name,split):
        """
        :param label_dir:
        :param visual_dir:
        :param audio_dir:
        :param tra_dir:
        :param labelgcn_name:
        :param split:
        """
        self.split = split
        self.visual = sio.loadmat(visual_dir)[self.split]
        self.audio = sio.loadmat(audio_dir)[self.split]
        self.tra = sio.loadmat(tra_dir)[self.split]
        self.label = sio.loadmat(label_dir)[self.split]
        self.labelgcn = sio.loadmat(labelgcn_name)['labelVector']

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self,idx):
        visual = self.visual[idx]
        audio = self.audio[idx]
        tra = self.tra[idx]
        label = self.label[idx]
        return visual, audio, tra, label, self.labelgcn


class VideoDatasetTokE(object):


    def __init__(self,label_dir,visual_dir,audio_dir,tra_dir,labelgcn_name,split):
        """
        :param label_dir:
        :param visual_dir:
        :param audio_dir:
        :param tra_dir:
        :param labelgcn_name:
        :param split:
        """
        self.split = split
        self.visual = sio.loadmat(visual_dir)[self.split]
        self.audio = sio.loadmat(audio_dir)[self.split]
        self.tra = sio.loadmat(tra_dir)[self.split]
        self.label = sio.loadmat(label_dir)[self.split]
        self.labelgcn = sio.loadmat(labelgcn_name)['labelVector']

    def cal_euclidean_matrix(self, mat):
        m = mat.shape[0]
        n = mat.shape[1]
        result =  np.sqrt(np.matmul(mat**2, np.ones((n, m)))+ \
            np.transpose(np.matmul(mat**2, np.ones((n, m)))) - \
                2*np.matmul(mat, np.transpose(mat)))
        return result

    def run(self):
        self.cal_euclidean_matrix(self.visual)