from django.conf import settings
from django.shortcuts import render
from datetime import datetime
from images.models import Image
from images.service import ImageService
import os, sys, gc

# Create your views here.
@csrf_exempt
def compute(request):
    if request.method != "POST":
        return render(request, "images/compute.html", locals())

    if 'image' in request.FILES:
        img_obj = request.FILES.get('image')
        img_name = 'temp/temp_file-%s.jpg' % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        img_full_path = os.path.join(settings.MEDIA_ROOT, img_name)
        dest = open(img_full_path, 'w')
        dest.write(img_obj.read())
        dest.close()
    try:
        gc.collect()
        link1, link2, link3 = ImageService.get_target(img_full_path)
    except:
        pass
    return render(request, "images/compute.html", locals())
