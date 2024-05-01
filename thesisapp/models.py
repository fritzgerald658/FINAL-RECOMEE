from django.db import models


class UserHistory(models.Model):
    username = models.CharField(max_length=100, null=True)
    user_course = models.CharField(max_length=100, null=True)
    user_skills = models.CharField(max_length=100, null=True)
    user_interest = models.CharField(max_length=100, null=True)
    user_industry = models.CharField(max_length=100, null=True)

class PredictionResult(models.Model):
    career_one = models.CharField(max_length=100, null=True)
    career_two = models.CharField(max_length=100, null=True)
    career_three = models.CharField(max_length=100, null=True)

    