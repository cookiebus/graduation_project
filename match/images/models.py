from django.db import models
from jsonfield import JSONField
import cv2
import json

# Create your models here.
class Image(models.Model):
    image = models.ImageField(blank=True, upload_to='media/images/problems')
    model_3D = models.ImageField(blank=True, upload_to='media/3D_models/problems')

    kp = JSONField(blank=True, null=True)
    des = JSONField(blank=True, null=True)

    create_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering = ('id', )

    def save(self, *args, **kwargs):
        img = cv2.imread(self.image)
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img, None)
        self.kp = json.dumps(kp)
        self.des = json.dumps(des)
        
        super(Image, self).save(*args, **kwargs)

    def __str__(self):
        return "(%s)" % self.id