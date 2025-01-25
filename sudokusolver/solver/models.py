from django.db import models

# Create your models here.
class Image(models.Model):
    time = models.TimeField(auto_now=True)
    image = models.ImageField(upload_to='images/')

    def __str__(self):
        return str(self.time)