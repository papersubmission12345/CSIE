import argparse, time, os
import imageio
import numpy as np
import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
import torch
from torch.autograd import Variable

def predictscore(listLRs, listSRs, LRpath_list):
    optpath = './options/test/testScorenet_load.json'
    opt = option.parse_imagetest(optpath)
    opt = option.dict_to_nonedict(opt)

    # initial configure
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()
    if opt['self_ensemble']: model_name += 'plus'

    tmpsolver = create_solver(opt)
    #print('===> Start Test')
    #print("==================================================")
    listscore = []
    
    for i in range(len(listLRs)):
        LR = listLRs[i]
        SR = listSRs[i]
        LR, SR = np2Tensor([LR,SR], opt['rgb_range'])
        tmpsolver.feed_imgs(LR,SR)
        tmpsolver.test_scorenet()
        print(LRpath_list[i])
        visuals = tmpsolver.get_current_visual_scorenet(need_HR=None)
        score = visuals['predQS']#.to(torch.double) 
        listscore.append(score)

    return listscore


def main():
    parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)
    opt = option.dict_to_nonedict(opt)

    # initial configure
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()
    if opt['self_ensemble']: model_name += 'plus'

    # create test dataloader
    bm_names =[]
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        print('===> Test Dataset: [%s]   Number of images: [%d]' % (test_set.name(), len(test_set)))
        bm_names.append(test_set.name())

    # create solver (and load model)
    solver = create_solver(opt)
    # Test phase
    print('===> Start Test')
    print("==================================================")
    print("Method: %s || Scale: %d || Degradation: %s"%(model_name, scale, degrad))

    f = open(opt['savefile'], "w")


    sr_list = []
    lr_list = []
    LRpath_list = []
    #ORpath_list = []

    need_HR = False if test_loader.dataset.__class__.__name__.find('LRHR') < 0 else True
    

    #save a txt file for score net include: 
    #### LR SR score for LRHR            
    #### LR SR for LR only  
    save_img_path = opt['dir']

    
    for iter, batch in enumerate(test_loader):
    
        solver.feed_data(batch, need_HR=need_HR)
        solver.test()

        visuals = solver.get_current_visual(need_HR=need_HR)
        sr_list.append(visuals['SR'])
        lr_list.append(visuals['LR'])
        print(batch['LR_path'])
        LRpath_list.append(batch['LR_path'])
        #ORpath_list.append(batch['HR_path'])
        
    listscores = predictscore(lr_list,sr_list,LRpath_list)
    for i in range(len(listscores)):
        #write name and score
        tmpscore = str(listscores[i])
        tmpscore = tmpscore.replace('tensor','')
        tmpscore = tmpscore.replace('[','')
        tmpscore = tmpscore.replace(']','')
        tmpscore = tmpscore.replace('(','')
        tmpscore = tmpscore.replace(')','')
        
        tmppath = str(LRpath_list[i])
        tmppath = tmppath.replace('\'','')
        tmppath = tmppath.replace('[','')
        tmppath = tmppath.replace(']','')
        writetofile = tmppath + '\t' + tmpscore + '\n'
        f.write(writetofile)
        
    f.close()    
    print("==================================================")
    print("===> Finished !")
    
def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        # if img.shape[2] == 3: # for opencv imread
        #     img = img[:, :, [2, 1, 0]]
        h,w,c = img.shape
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img_tensorshape = np.zeros((1, c, h, w))
        img_tensorshape[0,:,:,:]=np_transpose
        tensor = torch.from_numpy(img_tensorshape).float()
        tensor.mul_(rgb_range / 255.)

        return tensor

    return [_np2Tensor(_l) for _l in l]

if __name__ == '__main__':
    main()