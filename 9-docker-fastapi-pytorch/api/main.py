from fastapi import FastAPI, status, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from fastapi.openapi.docs import (
    get_swagger_ui_html,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms, models 

from PIL import Image, ImageOps
import base64
from io import BytesIO
import cv2

from iteround import saferound

# specify your API title
app = FastAPI(
        title="Your API",
        description="Your API: Documentation",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        )

# CORS definition
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Swagger docs
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title,
        swagger_favicon_url="/static/logo.png",
    )

# Auto redirect to the doc page
@app.get("/", include_in_schema=False)
async def root():
    response = RedirectResponse(url='/docs')
    return response

# Request body specification
class ImageData(BaseModel):
    img: str

    class Config:
        schema_extra = {
                "example": {
                            "img": "{replace this with a base64 string}"
                }
        }

# pretrained models

class FinetunedModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # load pretrained model
        model = models.alexnet(pretrained=True)
        
        # NEW
        # we need to tap into the layer before the max pool in the convoluational layer
        self.features = model.features[:12]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        
        # We don't need to freeze it as we will use our pretrained model
        """
        # freeze the feature learning
        for param in self.features.parameters():
              param.requires_grad = False
        """
        
        # Instead we need to temporarily save the gradient values
        self.gradients = None
        
        # change the number of output classes of the last layer
        # this is useless line as it the number of output classes is already set to be 10
        self.classifier[-1] = nn.Linear(
            in_features=self.classifier[-1].in_features,
            out_features=2)
        
        # follow https://pytorch.org/hub/pytorch_vision_alexnet/
        tf_resize = transforms.Resize((256,256)) 
        tf_centercrop = transforms.CenterCrop(224)
        tf_totensor = transforms.ToTensor() 
        tf_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        self.tf_compose = transforms.Compose([
            tf_resize,
            tf_centercrop,
            tf_totensor,
            tf_normalize,
        ])
        
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self,x):
        x = self.features(x)
    
        # register the hook
        h = x.register_hook(self.activations_hook)
    
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features(x)

def val_transforms():
    
    return transforms.Compose(
        [
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

model = FinetunedModel.load_from_checkpoint(
        checkpoint_path='app/epoch=4-step=115.ckpt',
    )

# Endpoint to perform an image recognition
@app.post("/image-recognition")
async def image_recognition(image_data: ImageData):

    try:
        img = base64.b64decode(str(image_data.img)) 

        img = Image.open(BytesIO(img)) 
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        transform = val_transforms()
        img_tensor = transform(img).unsqueeze(0)

        pred =  model.forward(img_tensor)

        # detect
        logits = model(img_tensor)
        cat_id = int(torch.argmax(logits))

        # GRAD-CAM
        # get the gradient of the output with respect to the parameters of the model
        logits[:,cat_id].backward()

        # pull the gradients out of the model
        gradients = model.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = model.get_activations(img_tensor).detach()

        # weight the channels by corresponding gradients
        for i in range(256):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        img = np.array(img) 
        heatmap_resized = cv2.resize(heatmap.detach().numpy(), (img.shape[1], img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_resized = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB)
        superimposed_img = np.uint8(heatmap_resized * 0.2 + 0.8*img)

        im = Image.fromarray(superimposed_img)
        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())

        dataset_classes = ['bus','car']
        
        probability = nn.functional.softmax(logits, dim=-1)[0].detach().numpy()
        prob_percentage = saferound((probability*100).tolist(), places=3)
        
        result_prob = [
            {
                "class": dataset_classes[i],
                "prob":prob_percentage[i]
            } for i in range(2)
        ]

        return {
            "success": True,
            "result":dataset_classes[pred.argmax()],
            "detail":result_prob,
            "gradcam": img_str
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR ,
            # detail=str(e),
        )
