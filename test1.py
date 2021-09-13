import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import math
import PySimpleGUI as sg
import os
import shutil
import sys, os
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
    
    
print(application_path)


sg.theme("DarkTeal2")
layout = [[sg.T("")], [sg.Text("Choose input folder: "), sg.FolderBrowse(key="-IN1-")],[sg.Text("Choose target folder: "), sg.FolderBrowse(key="-IN2-")],[sg.Button("Submit")]]

###Building Window
window = sg.Window('My File Browser', layout, size=(600,150))
dir1,dir2=None,None    
while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED or event=="Exit":
        break
    elif event == "Submit":
        dir1=values["-IN1-"]
        dir2=values["-IN2-"]
        break
window.close()        
       
model_path = application_path+'/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
#device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
device = torch.device('cpu')



model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)



for filename in os.listdir(dir1):
    
    os.makedirs(dir2+"/"+'test', exist_ok=True)
    img = cv2.imread(dir1+"/"+filename)

    img_shape = img.shape
    tile_size = (400, 400)
    offset = (400, 400)

    for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
            cropped_img = img[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
        
            cropped_img = cropped_img * 1.0 / 255
            cropped_img = torch.from_numpy(np.transpose(cropped_img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = cropped_img.unsqueeze(0)
            img_LR = img_LR.to(device)
            with torch.no_grad():
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round()
            cv2.imwrite(dir2+"/"+"test/debug_" + str(i) + "_" + str(j) + ".png", output)

    img=None
    result=None
    for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
        imgh=None
        for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
            curr_img=cv2.imread(dir2+"/"+"test/debug_" + str(i) + "_" + str(j) + ".png")
            if imgh is None:
                imgh=curr_img
            else:
                imgh= cv2.hconcat([imgh, curr_img])
        if result is None:
            result=imgh
        else:
            result=cv2.vconcat([result,imgh])
    cv2.imwrite(dir2+'/'+filename.split(".")[0]+'_output.png', result)
    shutil.rmtree(dir2+"/"+"test", ignore_errors=True)
    
