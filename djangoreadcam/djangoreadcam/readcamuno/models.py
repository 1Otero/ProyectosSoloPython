from django.db import models

# Create your models here.
class clien(models.Model):
    _id= models.TextField("id", primary_key=True, max_length=int(100))
    name= models.TextField("name", max_length=int(100))