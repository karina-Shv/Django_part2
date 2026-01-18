from django.urls import path
from . import views

app_name = "tf_tasks"

urlpatterns = [
    path("", views.index, name="index"),
    path("task1/", views.task1, name="task1"),
    path("task2/", views.task2, name="task2"),
    path("task3/", views.task3, name="task3"),
    path("task4/", views.task4, name="task4"),
    path("task5/", views.task5, name="task5"),
]