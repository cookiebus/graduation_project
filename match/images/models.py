from django.db import models
from jsonfield import JSONField
import cv2
import json

# Create your models here.
class Image(models.Model):
    image = models.ImageField(blank=True, upload_to='media/images')
    model_3D = models.ImageField(blank=True, upload_to='media/3D_models')

    kp = JSONField(blank=True, null=True)
    des = JSONField(blank=True, null=True)

    create_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering = ('id', )

    def __str__(self):
        return "(%s)" % self.id
