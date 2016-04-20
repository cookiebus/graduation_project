from django.shortcuts import render

# Create your views here.
def compute(request):
    images = Image.objects.all()
    for image in images:
        
