from django.db import models


# Create your models here.
class User(models.Model):
    username = models.CharField(max_length=30, unique=True)
    password = models.CharField(max_length=128)
    # why iris image? so that if we change normalize algorithm it works
    iris_image1 = models.ImageField(upload_to="iris/")
    iris_image2 = models.ImageField(upload_to="iris/")
    iris_image3 = models.ImageField(upload_to="iris/")

    file = models.FileField(upload_to="files/")

    def __str__(self):
        return self.username
