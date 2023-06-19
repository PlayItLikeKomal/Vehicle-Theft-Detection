import easyocr
import pandas as pd
import os 
import glob
import re
import cv2

def Excl(ocr_path):

  df = pd.DataFrame(columns=['Image_ID','Text'])
  ocr = easyocr.Reader(['en'],gpu=True)
  print('OCR Process Started!!')
  folder_path = ocr_path
  for i,path in enumerate(glob.glob(folder_path+'/*')):
    print(path)
    img = cv2.imread(path)
    result = ocr.readtext(img)  
    for (box,text,score) in result:
      if score > .2:
        text = re.sub("[.,:\[\];?/\$@^#!&*]",'',text)
        if len(text) < 6:
          continue
        df.loc[i,'Image_ID'] = path.split('/')[-1]
        df.loc[i,'Text'] = text.upper()
  df.to_csv("D:\\Final_Fyp\\yolov5\\ESRGAN\\NB_plate_SR.csv",mode='a',header=False,index=False)
  print('OCR Process Complete!!')