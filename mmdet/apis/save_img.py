import cv2
import numpy as np


def save_img(img_adv, boxes_batch, save_path):
    gg = img_adv.clone()
    save_img = gg[0].permute(1,2,0).squeeze(0).detach().cpu().numpy()[:,:,(2,1,0)].copy()
    box = boxes_batch[0]
    for b in box:
        y1, x1, y2, x2 = b
        cv2.line(save_img,(int(y1),int(x1)),(int(y1),int(x2)),(0,0,255),2)
        cv2.line(save_img,(int(y1),int(x1)),(int(y2),int(x1)),(0,0,255),2)
        cv2.line(save_img,(int(y2),int(x2)),(int(y1),int(x2)),(0,0,255),2)
        cv2.line(save_img,(int(y2),int(x2)),(int(y2),int(x1)),(0,0,255),2)
        cv2.imwrite(save_path, save_img)
    exit(0)