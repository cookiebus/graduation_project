from django.db import models


# Create your models here.
class Image(models.Model):
    image = models.ImageField(blank=True, upload_to='media/images')
    model_3D = models.ImageField(blank=True, upload_to='media/3D_models')

    create_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering = ('id', )

    def __str__(self):
        return "(%s)" % self.id
