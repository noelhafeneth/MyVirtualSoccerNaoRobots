from players.player import Player
import numpy as np
import cv2




class PlayerSimple(Player):

    #noball = State(initial=True)
    #ballfound = State()
    #ballfront = State()


    def __init__(self):
        Player.__init__(self)

    def detect_ball(self, img_seg):

        img_hsv = cv2.cvtColor(img_seg, cv2.COLOR_BGR2HSV)

        # Define the range of HSV (Hue, Saturation, Value)
        lower = np.array([0, 50, 50])  # 0 degree
        upper = np.array([10, 255, 255])  # 20 degree
        mask = cv2.inRange(img_hsv, lower, upper)

        if np.any(mask):
            y, x = np.where(mask)
            return [np.min(x), np.min(y), np.max(x), np.max(y)]

        else:
            print("None")
            return None


    def segment_image(self, img):
        x = np.float32(img.reshape((-1, 4)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 6  # number of clusters
        ret, label, center = cv2.kmeans(x, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        img_seg = center[label.flatten()]
        img_seg = img_seg.reshape((img.shape))
        return img_seg

    def behave(self):
        while True:
            img_bot = self.get_camera_frame('bottom')
            img_bot = cv2.cvtColor(img_bot, cv2.COLOR_RGBA2BGR)
            img_seg = self.segment_image(img_bot)
            cv2.imshow('Bot', img_seg)

            ball=self.detect_ball(img_seg)

            if(ball==None):
                self.move("TurnRight")

            else:
                print(f"{ball}")
                if ball[2]-ball[0]>100:
                    a=1
                elif ball[2]-ball[0]<5:
                    a=2
                elif 0.5*(ball[2]+ball[0]) < 60:
                    self.move("TurnLeft")
                elif 0.5*(ball[2]+ball[0]) > 100:
                    self.move("TurnRight")
                else:
                    self.move("Forwards")



            cv2.waitKey(1)

        return