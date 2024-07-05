import msr
import torch
import cv2
import numpy as np
from lightglue.utils import rbd, numpy_image_to_torch
import os
from os import path as osp

def Match(extractor, matcher, src_img_path, dst_img_path, out_img_path='', enhance=True, save=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
    #---------------------first step: input, enhance by traditional methods-------------------------------
    img0 = cv2.imread(src_img_path, cv2.IMREAD_COLOR)    #BGR    low_light
    img1 = cv2.imread(dst_img_path, cv2.IMREAD_COLOR)    #BGR    reference image
    if enhance:
        img0_light = msr.MSR(img0)      
        image0 = numpy_image_to_torch(img0_light[...,::-1])  #torch  0~1  low_light enhanced by retinex as input
    else:
        image0 = numpy_image_to_torch(img0[...,::-1])  #torch  0~1  low_light enhanced by retinex as input
    image1 = numpy_image_to_torch(img1[...,::-1])  #torch  0~1 
        
    #---------------------second step: match-------------------------------
    

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]  # the key points [len,2]

    if(len(m_kpts0)<20):
        print("{}匹配失败".format(src_img_path))
        return 0



    #---------------------third step: transform-------------------------------                                                     

    h, mask = cv2.findHomography(np.array(m_kpts0.cpu()), np.array(m_kpts1.cpu()), cv2.RANSAC)  # find the transform matrix
    height, width, channels = img1.shape
    im0Reg = cv2.warpPerspective(img0, h, (width, height))  #transform

    path_prefix = osp.commonprefix([src_img_path, dst_img_path])
    path_basename = osp.basename(src_img_path)
    out_img_path = osp.join(path_prefix, 'low_match', path_basename)
    print(out_img_path)
    if save:
        print("Saving matched image : ", out_img_path)
        with open(osp.join(path_prefix, 'num_match.txt'), 'a') as f:
            f.write(osp.splitext(path_basename)[0])
            f.write('\r\n')
        f.close()
        cv2.imwrite(out_img_path, im0Reg) # save the image
    return 1