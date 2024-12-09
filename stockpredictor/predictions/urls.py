from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_stock, name='predict_stock'),
    path('values/', views.prediction_values, name='prediction_values'),
]
