import numpy as np
from PIL import ImageGrab
import PIL
import cv2
import time
from matplotlib import pyplot as plt
import pyautogui
import imutils


class KirbybotModelSC2:
    def __init__(self):
        self.mini_map = np.array(ImageGrab.grab(bbox=(29, 820, 291, 1053)))
        self.unit_portraits = np.array(ImageGrab.grab(bbox=(662, 883, 1127, 1061)))
        self.resources = np.array(ImageGrab.grab(bbox=(1765, 18, 1860, 35)))
        self.minerals = np.array(ImageGrab.grab(bbox=(1520, 18, 1600, 35)))
        self.gas = np.array(ImageGrab.grab(bbox=(1640, 18, 1720, 35)))
        self.supply = np.array(ImageGrab.grab(bbox=(1765, 18, 1860, 35)))  # how much current supply
        self.total_supply = np.array(ImageGrab.grab(bbox=(1507, 15, 1840, 35)))  # curent and max
        self.time_box = np.array(ImageGrab.grab(bbox=(271, 778, 329, 798)))

    def eyes(self):
        " Loads all relevent screen coords using numpy arrays"
        cv2.imshow('Unit Portraits ', np.array(ImageGrab.grab(bbox=(662, 883, 1127, 1061))))
        cv2.imshow('Mini Map', np.array(ImageGrab.grab(bbox=(29, 820, 291, 1053))))
        cv2.imshow('Time', np.array(ImageGrab.grab(bbox=(271, 778, 329, 798))))
        cv2.imshow('Minerals', np.array(ImageGrab.grab(bbox=(1520, 18, 1600, 35))))
        cv2.imshow('Gas', np.array(ImageGrab.grab(bbox=(1640, 18, 1720, 35))))
        cv2.imshow('Supply', np.array(ImageGrab.grab(bbox=(1765, 18, 1860, 35))))

    def hatchcount(self):
        """determines if certain x, y positions are exactly the RGB value defined, if so, it updates basecnt"""
        screen = ImageGrab.grab()
        basecnt = 0
        if screen.load()[710, 1018] == (16, 169, 0):
            basecnt = 1
        elif screen.load()[773, 897] == (200, 114, 57):
            basecnt = 2
        elif screen.load()[832, 909] == (244, 139, 70):
            basecnt = 3
        elif screen.load()[889, 911] == (207, 118, 59):
            basecnt = 4
        elif screen.load()[947, 909] == (244, 139, 70):
            basecnt = 5
        return basecnt

    def queen_inject(self):
        """automates queens injects"""

        pyautogui.hotkey('ctrl', 'f1')
        pyautogui.hotkey('ctrl', '12')
        pyautogui.hotkey('5')
        pyautogui.hotkey('4')
        basecnt = self.hatchcount()
        print(basecnt)
        while basecnt >= 0:
            pyautogui.hotkey('backspace')
            pyautogui.keyDown('x')
            pyautogui.click(x=960, y=418)
            pyautogui.keyUp('x')
            basecnt -= 1
        pyautogui.hotkey('f1')
        pyautogui.hotkey('12')

    def unitctr(self):
        """Identifies unit port traits based on if the given pixel value is not black"""

        screen = ImageGrab.grab()
        all_positions = [[690, 913], [750, 913], [810, 913], [870, 913], [930, 913], [1050, 913], [1110, 913],
                         [690, 967], [750, 967], [810, 967], [870, 967], [930, 967], [1050, 967], [1110, 967],
                         [690, 1028], [750, 1028], [810, 1028], [870, 1028], [930, 1028], [1050, 1028], [1110, 1028]]

        return all_positions


    def unitcnt_tempmatch(self):
        '''finds the amount of units by template matching'''

        # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        ImageGrab.grab().save('main.jpg', "JPEG")

        screen = cv2.imread('main.jpg')
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('queen.jpg', 0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = np.where(res >= threshold)
        for i in res:
            print(cv2.minMaxLoc(i))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(screen, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
            print()
        cv2.imshow('Detected', screen)

    def unitcnt_cont(self):
            '''tries to count unit portraits using the contour method '''
            #img = cv2.imread('main.jpg')




            img = np.array(ImageGrab.grab(bbox=(662, 883, 1127, 1061)))



            blurred = cv2.pyrMeanShiftFiltering(img, 21, 51)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            #cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

            for c in cnts:
                try:
                    m = cv2.moments(c)
                    cx = int(m['m10'] / m['m00'])
                    cy = int(m['m01'] / m['m00'])
                    cv2.drawContours(img, c, -1, (0, 255, 0), 1)
                    cv2.circle(img, (cx, cy), 7, (255, 255, 255), 1)
                    cv2.putText(img, "center", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    #cv2.imshow('yes', np.array(img))
                except ZeroDivisionError:
                    continue
            cv2.imshow('yes', img)
            cv2.imshow('Blurred', blurred)
            cv2.imshow("threshold", thresh)



    def process_img(self, original_image):
        processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)  # Converted to gray
        # processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300) # edge detection
        return processed_img


win = KirbybotModelSC2()
#win.unitcnt_cont()

while True:
    # win = KirbybotModelSC2()
    # screen = ImageGrab.grab()
    # win.queen_inject()
    time.sleep(2)
    win.unitcnt_cont()

    cv2.waitKey()

    #win.unitctr()
    # print(win.unitctr())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # win.queen_inject()
    # print("Waiting 30 seconds")
    # time.sleep(15)
    # print('15 seconds')
    # time.sleep(10)
    # print('5 seconds')
    # time.sleep(5)
    # print(win.hatchcount())
    # x, y = pyautogui.position()
    # print()
    # print(x)
    # print(y)
    # time.sleep(2)


# screen.release()
