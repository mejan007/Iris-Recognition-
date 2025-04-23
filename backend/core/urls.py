from django.urls import path

from . import views  # Assuming your views are in the same app directory

urlpatterns = [
    path("", views.home_page, name="home"),
    path("register/", views.register, name="register"),
    path("login/", views.login, name="login"),
]
