from lightglue import LightGlue, SuperPoint, DISK
import torch
from match import Match
#import sys
#sys.path.append('./')

torch.set_grad_enabled(False)
path = "assets"   # the path of images
#path = "../Datasets/tieta/train/0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device) #load the match model

Match(extractor=extractor, matcher=matcher, 
      src_img_path=path + "/sacre_coeur1.jpg", 
      dst_img_path=path + "/sacre_coeur2.jpg", 
      out_img_path=path+"/sacre_coeur_match.png")

