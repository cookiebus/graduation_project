from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


# Create your views here.
def JsonResponse(params):
    return HttpResponse(json.dumps(params))


@csrf_exempt
def upload_image(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Please send a post."})

    if 'image' in request.FILES:
        file_obj = request.FILES.get('image')
        file_name = 'media/upload/temp_file-%d.jpg' % random.randint(0,10000000)
        file_full_path = os.path.join(settings.MEDIA_ROOT, file_name)
        dest = open(file_full_path, 'w')
        dest.write(file_obj.read())
        dest.close()
    else:
        file_full_path = ''

    return JsonResponse({"success": True, "image_path": file_full_path})
