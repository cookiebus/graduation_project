from images.models import Image
import cv2
import json

class ImageService(object):

    @classmethod
    def get_target(cls, img_path):
        sift = cv2.xfeatures2d.SIFT_create()
        img = cv2.imread(img_path)
        kp, des = sift.detectAndCompute(img, None)
        print kp
        print des
        return Image.objects.all()[0].image.url
