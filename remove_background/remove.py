import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import pyautogui

filelist = ["test1.png", "test2.png", "test3.png", "test4.png", "test6.png", "test7.png"]

height = pyautogui.size()[1]

y = 0
x = 0
for file in filelist:
    print(file)
    
    img = cv2.imread(file)
    scale = 0.4
    dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Input_" + file, img)
    cv2.moveWindow("Input_" + file, x, y)

    segmentor = SelfiSegmentation()
    imgout = segmentor.removeBG(img, (0,0,255), threshold=.95)
    cv2.imshow("Output_"+file, imgout)
    cv2.moveWindow("Output_"+file, x+img.shape[1], y)

    y += img.shape[0] + 20
    if y + img.shape[1] > height:
        y = 0
        x += img.shape[1]*2

    cv2.waitKey(0)
