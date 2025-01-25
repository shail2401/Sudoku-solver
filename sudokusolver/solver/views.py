from django.http.response import HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from .forms import *
from .models import Image

# Create your views here.
def index(request):
    if request.method == 'POST':
        form = ImageUpload(request.POST, request.FILES)
        if form.is_valid():
            time = form.cleaned_data.get('time')
            image = form.cleaned_data.get('image')
            imgo = Image.objects.create(
                time = time,
                image = image
            )
            imgo.save()
        return HttpResponseRedirect(reverse("solver:index"))
    else:
        return render(request, "solver/index.html",{
            'form': ImageUpload()
        })
