# MSR
import numpy as np
import cv2


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data
 
 
def SR(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)
 
    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)
 
    dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8
 
def MSR(img):
    scales = [15, 101, 301] #可调整的位置
    print(img.shape)
    b_gray, g_gray, r_gray = cv2.split(img)
    b_gray = SR(b_gray, scales)
    g_gray = SR(g_gray, scales)
    r_gray = SR(r_gray, scales)
    result = cv2.merge([b_gray, g_gray, r_gray])
    return result
    
    
    
if __name__ == '__main__':
    img = '../Datasets/LOL/eval15/low/1.png'
    scales = [15, 101, 301] #可调整的位置
    src_img = cv2.imread(img)
    b_gray, g_gray, r_gray = cv2.split(src_img)
    b_gray = MSR(b_gray, scales)
    g_gray = MSR(g_gray, scales)
    r_gray = MSR(r_gray, scales)
    result = cv2.merge([b_gray, g_gray, r_gray])
    #utils.save_img('1.png', result)
    cv2.imwrite('1.png', result)
    #cv2.imshow('Original Image', src_img)
    #cv2.imshow('Enhanced(MSR) Image', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()