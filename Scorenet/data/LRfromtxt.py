import torch.utils.data as data

from data import common


class LRfromtxt(data.Dataset):
    '''
    Read LR images only in test phase.
    '''

    def name(self):
        return common.find_benchmark(self.opt['txtpath'])


    def __init__(self, opt):
        super(LRfromtxt, self).__init__()
        self.opt = opt
        self.scale = self.opt['scale']
        self.paths_LR = None
        self.paths_HR = None
        self.scores = None
        self.list_LR = []
        
        # read image list from image/binary files
        self.paths_LR, self.paths_HR, self.scores = common.get_image_paths_from_txt(self.opt['txtpath'],0)
        
        for i in range(len(self.paths_LR)):
            lr_path = self.paths_LR[i]
            lr = common.read_img_fromtxt(lr_path, self.opt['data_type'])
            self.list_LR.append(lr) 
        
        assert self.paths_LR, '[Error] LR paths are empty.'


    def __getitem__(self, idx):
        # get LR image
        lr=  self.list_LR[idx]
        lr_path = self.paths_LR[idx]
        lr_tensor = common.np2Tensor([lr], self.opt['rgb_range'])[0]
        return {'LR': lr_tensor, 'LR_path': lr_path}


    def __len__(self):
        return len(self.paths_LR)