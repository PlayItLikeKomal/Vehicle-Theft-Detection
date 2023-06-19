import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import ESRGAN.RRDBNet_arch as arch
import re
#from ESRGAN.OCR import Excl
from datetime import datetime
def ESR(exp):
    i=0
    det_date = datetime.now().strftime("%Y_%m_%d")
    esr_path = 'D:\\Final_Fyp\\yolov5\\ESRGAN\\results\\'+det_date
    #ext_path = esr_path+'\\'+os.mkdir(exp)
    #if not os.path.exists(esr_path):
        #os.makedirs(esr_path)
        #esr_exp= os.makedirs(esr_path+'\\det{}'.format(str(i)))
        #cv2.imwrite(str(esr_exp)+'\\{:s}_rlt.png'.format(base), output)  #Number_Plate_detection_Yolov5-DeepSort/SR-Output
    #elif os.path.exists(esr_path+'\\det{}'.format(str(i))):
        #i=i+1
        #esr_exp= os.makedirs(esr_path+'\\det{}'.format(str(i)))    
        #cv2.imwrite(str(esr_exp)+'\\{:s}_rlt.png'.format(base), output)  #Number_Plate_detection_Yolov5-DeepSort/SR-Output
    #else:
        #esr_exp= os.makedirs(esr_path+'\\det{}'.format(str(i)))    
        #cv2.imwrite(str(esr_exp)+'\\{:s}_rlt.png'.format(base), output)  #Number_Plate_detection_Yolov5-DeepSort/SR-Output
 
    #i=i+1
    model_path =   'D:\\Final_Fyp\\yolov5\\ESRGAN\\models\\RRDB_ESRGAN_x4.pth'  # 'models/RRDB_PSNR_x4.pth' # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
    # RRDB_PSNR_x4
    device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
    device1 = torch.device('cpu')

    test_img_folder = exp+"\\crops\\license\\*"
    #'C:\\Final_Fyp\\yolov5\\runs\\detect\\exp{}\\crops\\license\\*'.format(i)
    torch.cuda.empty_cache() # Custom Clearing Cache
    # torch.cuda.set_per_process_memory_fraction(0.5, 0) # Custom Clearing Cache
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    #counter = 1
    print('Model path {:s}. \nTesting...'.format(model_path))
    print('ESR Process Started!!')
    idx = 0
    for it_path in glob.glob(test_img_folder):
        # regularExp = re.search('.[mp4]',path)
        # if regularExp.start():
        #     counter = 0
        # else:
        #     counter = 1
        # if counter == 1:
        # Custom Test
        idx += 1
        base = osp.splitext(osp.basename(it_path))[0]
        print(idx, base)
        # read images
        img = cv2.imread(it_path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        #cv2.imwrite(str(esr_exp)+'\\{:s}_rlt.png'.format(base), output)  #Number_Plate_detection_Yolov5-DeepSort/SR-Output
        if not os.path.exists(esr_path):
           os.makedirs(esr_path)
          #  esr_exp= os.makedirs(esr_path+'\\det{}'.format(str(i)))
           cv2.imwrite(str(esr_path)+'\\esr\\{:s}_rlt.png'.format(base), output)  #Number_Plate_detection_Yolov5-DeepSort/SR-Output
        else:
           cv2.imwrite(str(esr_path)+'\\esr\\{:s}_rlt.png'.format(base), output)  #Number_Plate_detection_Yolov5-DeepSort/SR-Output
            
        #elif os.path.exists(esr_path+'\\det{}'.format(str(i))):
         #   i=i+1
          #  esr_exp= os.makedirs(esr_path+'\\det{}'.format(str(i)))    
           # cv2.imwrite(str(esr_exp)+'\\{:s}_rlt.png'.format(base), output)  #Number_Plate_detection_Yolov5-DeepSort/SR-Output
        #else:
         #   esr_exp= os.makedirs(esr_path+'\\det{}'.format(str(i)))    
          #  cv2.imwrite(str(esr_exp)+'\\{:s}_rlt.png'.format(base), output)  #Number_Plate_detection_Yolov5-DeepSort/SR-Output
        # cv2.imwrite('Number_Plate_detection_Yolov5-DeepSort/SR-Output/{:s}_rlt.png'.format(base), output)"""        
        #Excl()
        #torch.cuda.empty_cache()
    print('ESR Process Completed!!')

