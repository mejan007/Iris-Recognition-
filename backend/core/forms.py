from django import forms

# from django.core.files.base import ContentFile
from django.core.validators import FileExtensionValidator
from .models import User
from .utils import (
    normalize_enchance_image,
    load_and_preprocess_image,
    check_password,
    custom_hash,
)
from django.conf import settings

# from django.contrib.auth.hashers import make_password


class RegisterForm(forms.Form):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)
    password1 = forms.CharField(widget=forms.PasswordInput)
    iris_image1 = forms.FileField(
        required=True,
        validators=[
            FileExtensionValidator(allowed_extensions=["jpg", "jpeg", "png", "bmp"])
        ],
    )
    iris_image2 = forms.FileField(
        required=True,
        validators=[
            FileExtensionValidator(allowed_extensions=["jpg", "jpeg", "png", "bmp"])
        ],
    )
    iris_image3 = forms.FileField(
        required=True,
        validators=[
            FileExtensionValidator(allowed_extensions=["jpg", "jpeg", "png", "bmp"])
        ],
    )
    file = forms.FileField(required=True)

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password1 = cleaned_data.get("password1")
        username = cleaned_data.get("username")
        if not password or not check_password(password):
            raise forms.ValidationError(
                "Passwords must contain one symbol, one capital letter and one number"
            )
        if password and password1 and password != password1:
            raise forms.ValidationError("Passwords dont match")
        try:
            user = User.objects.get(username=username)
            raise forms.ValidationError("Username exists")

        except User.DoesNotExist:
            # to test if images ar valid
            try:
                _, _, img1 = normalize_enchance_image(cleaned_data.get("iris_image1"))
            except ValueError:
                raise forms.ValidationError("Iris 1 couldn't be used")
            try:
                _, _, img2 = normalize_enchance_image(cleaned_data.get("iris_image2"))
            except ValueError:
                raise forms.ValidationError("Iris 2 couldn't be used")
            try:
                _, _, img3 = normalize_enchance_image(cleaned_data.get("iris_image3"))
            except ValueError:
                raise forms.ValidationError("Iris 3 couldn't be used")
            predictions = []
            img1 = load_and_preprocess_image(img1)
            img2 = load_and_preprocess_image(img2)
            img3 = load_and_preprocess_image(img3)
            predictions.append(settings.MODEL.predict([img1, img2]))
            predictions.append(settings.MODEL.predict([img1, img3]))
            predictions.append(settings.MODEL.predict([img2, img3]))

            for prediction in predictions:
                if prediction < 0.5:
                    raise forms.ValidationError("Images are not of the same iris")

            return cleaned_data

    def save(self, request):
        self.cleaned_data.pop("password1")
        self.cleaned_data["password"] = custom_hash(self.cleaned_data["password"])
        user = User.objects.create(**self.cleaned_data)
        user.save()
        return user


class LoginForm(forms.Form):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)
    iris_image = forms.FileField(
        required=True,
        validators=[
            FileExtensionValidator(allowed_extensions=["jpg", "jpeg", "png", "bmp"])
        ],
    )

    def clean_username(self):
        username = self.cleaned_data["username"]
        if len(username) < 3:
            raise forms.ValidationError("Username must be at least 3 characters long.")
        return username

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get("username")
        password = cleaned_data.get("password")
        user = User.objects.filter(username=username).first()
        if not user:
            raise forms.ValidationError("Invalid username")
        if not password or not check_password(password):
            raise forms.ValidationError(
                "Passwords must contain one symbol, one capital letter and one number"
            )
        if user.password != custom_hash(password):
            raise forms.ValidationError("Invalid password")
        _, _, img1 = normalize_enchance_image(user.iris_image1.path)
        _, _, img2 = normalize_enchance_image(user.iris_image2.path)
        _, _, img3 = normalize_enchance_image(user.iris_image3.path)
        usr_img = cleaned_data.get("iris_image")
        try:
            _, _, usr_img = normalize_enchance_image(cleaned_data.get("iris_image"))
        except ValueError:
            raise forms.ValidationError("Iris couldn't be used")
        img1 = load_and_preprocess_image(img1)
        img2 = load_and_preprocess_image(img2)
        img3 = load_and_preprocess_image(img3)
        usr_img = load_and_preprocess_image(usr_img)
        predictions = []
        predictions.append(settings.MODEL.predict([usr_img, img1]))
        predictions.append(settings.MODEL.predict([usr_img, img2]))
        predictions.append(settings.MODEL.predict([usr_img, img3]))
        avg = 0
        for i in range(3):
            avg += predictions[i]
        avg = avg / 3
        if avg < 0.5:
            raise forms.ValidationError("Iris doesnt match")
        return cleaned_data

    def save(self, request):
        cleaned_data = super().clean()
        username = cleaned_data.get("username")
        user = User.objects.filter(username=username).first()
        return user
