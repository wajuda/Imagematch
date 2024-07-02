#from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
import msr
import torch
import cv2
import numpy as np
#import sys
#sys.path.append('./')

torch.set_grad_enabled(False)
path = "assets"   # the path of images
#path = "../Datasets/tieta/train/0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device) #load the match model

#---------------------first step: enhance by traditional methods-------------------------------
img0 = cv2.imread(path + "/sacre_coeur1.jpg", cv2.IMREAD_COLOR)    #BGR    low_light
img1 = cv2.imread(path + "/sacre_coeur2.jpg", cv2.IMREAD_COLOR)    #BGR    reference image
img0_light = msr.MSR(img0)        # low_light enhanced by retinex 

'''
img0 = cv2.imread(path + "/0a.jpg", cv2.IMREAD_COLOR)    #BGR
img1 = cv2.imread(path + "/0b.png", cv2.IMREAD_COLOR)    #BGR    reference image
'''
#---------------------second step: match-------------------------------
image0 = numpy_image_to_torch(img0_light[...,::-1])  #torch  0~1  low_light enhanced by retinex as input
image1 = numpy_image_to_torch(img1[...,::-1])  #torch  0~1

feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]  # the key points [len,2]

if(len(m_kpts0)<20):
    print(m_kpts0.shape)
    raise ValueError("匹配可能失败，特征点较少")   # in case of error due to the awful quality of tieta


'''
axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers') 

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)     #plot the points match image
'''
#---------------------third step: transform-------------------------------                                                     

h, mask = cv2.findHomography(np.array(m_kpts0.cpu()), np.array(m_kpts1.cpu()), cv2.RANSAC)  # find the transform matrix
height, width, channels = img1.shape
im0Reg = cv2.warpPerspective(img0, h, (width, height))  #transform

outFilename = path+"/sacre_coeur_match.png"
print("Saving matched image : ", outFilename)
cv2.imwrite(outFilename, im0Reg) # save the image