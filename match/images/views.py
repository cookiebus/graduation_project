from django.shortcuts import render
import cv2
import json

# Create your views here.
def compute(request):
    images = Image.objects.all()
    sift = cv2.xfeatures2d.SIFT_create()

    for image in images:
        if not image.kp and not image.des:
            img = imread(image.image.url)
            kp, des = sift.detectAndCompute(img, None)
            image.kp = json.dumps(kp)
            image.des = json.dumps(des)
            image.save()

    return render(request, "images/compute.html", locals())