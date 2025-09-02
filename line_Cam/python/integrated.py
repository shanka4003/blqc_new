# main.py

from multiprocessing import Process
from OpenCV_Demo import gige_camera_worker
from Acquisition_pyspin import usb_camera_worker

if __name__ == '__main__':
    p1 = Process(target=gige_camera_worker)
    p2 = Process(target=usb_camera_worker)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
