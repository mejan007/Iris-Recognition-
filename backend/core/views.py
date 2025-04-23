from django.shortcuts import render, redirect
from .forms import RegisterForm
from .forms import LoginForm


# Create your views here.
def home_page(request):
    context = {}
    errors = request.session.get("errors", {})
    if errors:
        if "__all__" in errors:
            errors["credentials"] = errors["__all__"]  # Copy data to 'credentials' key
            errors.pop("__all__")  # Remove the '__all__' key
            context["errors"] = errors  # Update context with modified errors

            del request.session["errors"]

    return render(request, "main/index.html", context)


def register(request):
    context = {}
    errors = request.session.get("errors", {})
    if errors:
        if "__all__" in errors:
            errors["credentials"] = errors["__all__"]  # Copy data to 'credentials' key
            errors.pop("__all__")  # Remove the '__all__' key
            context["errors"] = errors  # Update context with modified errors

            del request.session["errors"]

    if request.method == "POST":
        form = RegisterForm(request.POST, request.FILES)
        if form.is_valid():
            form.save(request)
            return redirect("home")  # Redirect to your desired page
        else:
            request.session["errors"] = form.errors
            return redirect("register")
    else:
        form = RegisterForm()

    return render(request, "main/register.html", context=context)


def login(request):
    if request.method == "POST":
        form = LoginForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save(request)
            file = user.file
            return redirect(f"/media/{file}")  # Replace with your desired URL
        else:
            # Pass the form with errors to the template
            request.session["errors"] = form.errors
            return redirect("home")
            # return render(request, "main/login.html", {"form": form})

    else:
        form = LoginForm()

    return render(request, "main/login.html", {"form": form})
