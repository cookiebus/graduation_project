from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, HttpResponse
from django.conf import settings
from images.service import ImageService
import json
import os
import random
import gc
from PIL import Image


# Create your views here.
def JsonResponse(params):
    return HttpResponse(json.dumps(params))


@csrf_exempt
def upload_image(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Please send a post."})

    if 'image' in request.FILES:
        file_obj = request.FILES.get('image', '')
        file_name = 'upload/temp_file-%d.jpg' % random.randint(0,10000000)
        file_full_path = os.path.join(settings.MEDIA_ROOT, file_name)
        dest = open(file_full_path, 'w')
        dest.write(file_obj.read())
        dest.close()
        m1 = Image.open(file_full_path)
        w, h = m1.size
        m1 = m1.resize((w / 2, h / 2), Image.ANTIALIAS)
        m1 = m1.rotate(-90)
        m1.save(file_full_path)
    else:
        file_full_path = ''
        return JsonResponse({"success": False, "error": "Post image fail"})


    gc.collect()
    link1, link2, link3 = ImageService.get_target(file_full_path)
    print link1, link2, link3
    if link1.startswith("/media/"):
        file_name = link1[len('/media/'):]
        print file_name
        return JsonResponse({"success": True, "image_path": file_name})
    else:
        file_name = ''
        return JsonResponse({"success": False, "error": "Match image fail"})
