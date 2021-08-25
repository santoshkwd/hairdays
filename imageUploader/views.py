import time
import os
import json
import streamlit as st
import torch
from fastai.vision import *
import pandas as pd
import time

import PIL
from django.shortcuts import render, HttpResponse
import streamlit as st

# Create your views here.
from fastcore.basics import defaults
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage

@csrf_exempt
def home(request):
    result = ""
    if request.method == 'POST':
        uploaded_file = request.FILES['hairfile']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)

        img_obj = open_image("media/"+name, "r+")
        result, prob = processing(img_obj)

        rimages = "static/hairs/"+str(result)+".jpg";
        return render(request, 'indexs.html', {'result': result,'rimages':rimages,'img_obj': "media/"+name})

    return render(request, 'indexs.html', {'result': result,})


def result(request):
    return render(request, 'result.html')


def predict_hair_type(uploaded_file, model):
    defaults.device = torch.device('cpu')
    pred, pred_label, pred_prob = model.predict(uploaded_file)
    return (pred, pred_prob)


def processing(uploaded_file):
    with open('imageUploader/app_data/hair_type_desc.json', mode='r', encoding='utf-8') as f:
     hair_type_desc = json.load(f)
    path = os.getcwd() + '/imageUploader/model'
    # Load model
    model = load_learner(path, 'hair-cam-resnet.pkl')
    label, prob = predict_hair_type(uploaded_file, model)  # Make prediction
    hair_types = ['1', '2A', '2B', '2C', '3A', '3B',
                  '3C', '4A', '4B', '4C']
    h_type = hair_types[int(label)]

    return h_type, prob


