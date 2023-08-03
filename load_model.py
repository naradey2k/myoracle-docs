import os
import timm
import requests
import torch
import requests
import streamlit as st
import torchvision.transforms as transforms

from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import ViltForQuestionAnswering

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VQA_URL = "https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt"

@st.cache_resource
def load_model():
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    return (processor, model)

def load_img_model(name):
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

    return model, 'vitb16'

def label_count_list(labels):
    res = {}
    keys = set(labels)
    for key in keys:
        res[key] = labels.count(key)

    return res

def get_item(image, question, tokenizer, image_model, model_name):
    inputs = tokenizer(
        question,
        return_tensors='pt'
    )
    visual_embeds = get_img_feats(image, image_model=image_model, name=model_name)\
        .squeeze(2, 3).unsqueeze(0)

    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
    upd_dict = {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
    inputs.update(upd_dict)
   
    return upd_dict, inputs


def get_img_feats(image, image_model, new_size=None):
    if new_size is not None:
        transfrom_f = transforms.Resize((new_size, new_size), interpolation=transforms.InterpolationMode.LANCZOS)
        image = transfrom_f(image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image_features = image_model.forward_features(image.unsqueeze(0))

    return image_features


def get_data(query, delim=","):
    assert isinstance(query, str)
    if os.path.isfile(query):
        with open(query) as f:
            data = eval(f.read())
    else:
        req = requests.get(query)

        try:
            data = requests.json()

        except Exception:
            data = req.content.decode()
            assert data is not None, "could not connect"

            try:
                data = eval(data)
            except Exception:
                data = data.split("\n")

        req.close()

    return data

def err_msg():
    print("Load error, try again")

    return "[ERROR]"


def get_answer(processor, model, img, question):
    encoding = processor(images=img, text=question, return_tensors="pt")
    
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    pred = model.config.id2label[idx]

    return pred