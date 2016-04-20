from django.contrib import admin

# Register your models here.
from django.contrib import admin
from images.models import Image

# Register your models here.
class ImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'image', 'model_3D')
    ordering = ('-id', )


admin.site.register(Image, ImageAdmin)
