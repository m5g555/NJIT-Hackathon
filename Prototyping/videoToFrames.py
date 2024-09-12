import cv2 as cv
import os

def main(vidPath, targetFPS):
    vid = cv.VideoCapture(vidPath)
    
    if not os.path.exists("frames"):
        os.makedirs("frames")
        
    curFrame = 0
    
    
    while (True):
        for i in range(0,int(30/targetFPS)):
            ret, frame = vid.read()
        if ret:
            imgName = ".\\frames\\frame"+str(curFrame)+".jpg"
            print("creating image "+str(curFrame))
            cv.imwrite(imgName, frame)
            curFrame += int(30/(targetFPS))
        else:
            break
    
    vid.release()
    cv.destroyAllWindows()
        
main(".\\Prototyping\\Learn the Disco-Bedience.mp4", 30)