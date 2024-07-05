from lightglue import LightGlue, SuperPoint, DISK
import torch
from match import Match
import cv2
from data import paired_paths_from_folder
cv2.setNumThreads(1)
#import sys
#sys.path.append('./')

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device) #load the match model


lq_folder = "../../../../Datasets/tieta1/Test/low"
gt_folder = "../../../../Datasets/tieta1/Test/high"
paths = paired_paths_from_folder(folders=[lq_folder, gt_folder], keys= ['lq', 'gt'], filename_tmpl='{}')
num = 0
total = len(paths)
for path in paths:
    num = num +Match(extractor=extractor, matcher=matcher, 
    src_img_path=path['lq_path'], 
    dst_img_path=path['gt_path'], 
    #out_img_path=path,
    save=True)
print('成功率:{}'.format(num/total))
