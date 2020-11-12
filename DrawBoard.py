import cv2
import numpy as np

class DrawBoard:
    def __init__(self, generator):
        self.img = np.zeros((64,64,3), np.uint8)
        self.generator = generator
        self.lastx = -1
        self.lasty = -1
        self.drawing = False

        cv2.namedWindow('drawingboard')
        cv2.namedWindow('output')
        cv2.setMouseCallback('drawingboard', self.DrawLine)

    def Update(self):
        cv2.imshow('drawingboard', self.img)
        edge = np.array(self.img)
        edge = (edge - 127.5) / 127.5
        fake = self.generator.predict(edge.reshape(-1,64,64,3))
        fake = fake[0]
        fake = (fake + 1) / 2.0
        fake = cv2.cvtColor(fake,cv2.COLOR_BGR2RGB)
        cv2.imshow('output', fake)
        cv2.waitKey(1)
    
    def DrawLine(self,event,x,y,flags,param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.img = np.zeros((64,64,3), np.uint8)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.lastx = x
            self.lasty = y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing == True:
            cv2.line(self.img, (self.lastx,self.lasty), (x,y), (255,255,255), 2)
            #cv2.circle(img, (x, y), 1, (255,255,255), -1)
            self.lastx = x
            self.lasty = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False