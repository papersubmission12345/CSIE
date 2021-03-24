import torch.utils.data as data

from data import common


class LRHRfromtxt(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return common.find_benchmark(self.opt['txtpath'])


    def __init__(self, opt):
        super(LRHRfromtxt, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR, self.paths_LR = None, None
        self.qualityscore = None
        self.list_LR = []
        self.list_HR = []
        
        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 2

        # read image list from image/binary files
        self.paths_LR, self.paths_HR, self.qualityscore = common.get_image_paths_from_txt(self.opt['txtpath'],1)
        
        for i in range(len(self.paths_LR)):
            lr_path = self.paths_LR[i]
            hr_path = self.paths_HR[i]
            lr = common.read_img_fromtxt(lr_path, self.opt['data_type'])
            hr = common.read_img_fromtxt(hr_path, self.opt['data_type'])
            self.list_LR.append(lr) 
            self.list_HR.append(hr) 
        
        assert self.paths_HR, '[Error] HR paths are empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                '[Error] HR: [%d] and LR: [%d] have different number of images.'%(
                len(self.paths_LR), len(self.paths_HR))


    def __getitem__(self, idx):
       
        if self.train:
            i = idx % len(self.list_HR)
            lr      = self.list_LR[i]
            hr      = self.list_HR[i]
            
            lr, hr = self._get_patch(lr, hr)
            lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.opt['rgb_range'])
            return {'LR': lr_tensor, 'HR': hr_tensor}
            
        else:
            lr = self.list_LR[idx]
            hr = self.list_HR[idx]
            qualityscore = self.qualityscore[idx]
            lr_path = self.paths_LR[idx]
            hr_path = self.paths_HR[idx]
            lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.opt['rgb_range'])
            
            return {'LR': lr_tensor, 'HR': hr_tensor, 'QS':qualityscore, 'LR_path': lr_path, 'HR_path': hr_path}


    def __len__(self):
        #print(len(self.paths_HR) * self.repeat)
        if self.train:
            return len(self.paths_HR) * self.repeat
        else:
            return len(self.paths_LR)


    #for test or val
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.list_HR)
        else:
            return idx 
    '''
    #for test or val
    def _load_file(self, idx):
        print('_load_file')
        idx = self._get_index(idx)
        lr_path = self.paths_LR[idx]
        hr_path = self.paths_HR[idx]
        lr = common.read_img_fromtxt(lr_path, self.opt['data_type'])
        hr = common.read_img_fromtxt(hr_path, self.opt['data_type'])
        qc = self.qualityscore[idx]

        return lr, hr, qc, lr_path, hr_path
    '''

    def _get_patch(self, lr, hr):

        LR_size = self.opt['LR_size']
        
        lr, hr = common.get_patch(lr, hr, LR_size, self.scale)
        lr, hr = common.augment([lr, hr])

        return lr, hr

    def _get_patch_bk(self, lr, hr):
        LR_size = self.opt['LR_size']
        
        lr, hr = common.get_patch(lr, hr, LR_size, self.scale)
        lr, hr = common.augment([lr, hr])

        return lr, hr
