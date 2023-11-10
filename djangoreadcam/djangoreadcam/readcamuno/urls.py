from django.urls import path
from readcamuno.views import client, user, guess


urlpatterns= [
    path('client/', client, name='otroclient'),
    path('user/', guess, name='otrouser')
]