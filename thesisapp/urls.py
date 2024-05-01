from django.urls import path
from . import views

urlpatterns = [
    path('', views.get_started, name= 'get_started'),
    path('predict_position/', views.predict_position, name='predict_position'),
    path('display_data/', views.display_data, name='display_data'),
    path('no-results/', views.no_results, name='no_results')
   
]