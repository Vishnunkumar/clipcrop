# clipcrop
- Extract sections of images from your image by using OpenAI's CLIP and Facebooks Detr implemented on HuggingFace Transformers
- Added new capability for segmentation using CLIP and Detr segmentation models

# Why Detr?

Facebook's Detr is one of most effective object detection algorithm developed in the recent years. It simply expands to Detection Transformers and effectively a CNN architecture followed by Transformers encoders and decoders. It uses biopartite matching loss to compare objects detected in an image and reasons the predictions with the global image. Images are processed via CNN and encoder layer to output hidden states [number of images, seq_length, d_model] and object_queries [number of images, num of objects, d_model] are sent through decoders to get the neccessary logits for classification and MLP for regression(bounding box) Below are reason why you should prefer Detr over some popular algorithms

- It's single step detector and it's efficiency is on par and better than two stage detectors like RCNN and Fast RCNN.
- Compared to Yolo and SSD which are one stage detector DeTr performs detection on the whole image rather than grids of images unlike what we see in Yolo.

# Installation
```python
pip install clipcrop
```

## Clip Crop

Extract sections of images from your image by using OpenAI's CLIP and Facebooks Detr implemented on HuggingFace Transformers (Inspired from [@vijishmadhavan](https://github.com/vijishmadhavan/Crop-CLIP/))

### Implementation

```python
from clipcrop import clipcrop
clipc = clipcrop.ClipCrop("/content/nm.jpg", "woman in white frock")
DFE, DM, CLIPM, CLIPP = clipc.load_models()
result = clipc.extract_image(DFE, DM, CLIPM, CLIPP)
# gives a list of dicitonary of top images and its relative similarity score and you can override this by setting num = 5  to get top 5 etc while initiating the class
```
<!-- 
### Result

<p style="font-style: italic;">clipcrop = ClipCrop("/content/nm.jpg", "woman in white frock")</p>
<p float="left">
<img src="/nm.jpg" width="600" height="350">
<img src="/clipcrop.jpeg" width="150" height="300">
</p>

<br>

<p style="font-style: italic;">cc = ClipCrop('/content/rd.jpg', 'woman walking', 2)</p>
<p float="left">
<img src="/rd.jpg" width="600" height="350">
<img src="/rmc.jpeg" width="150" height="300">
</p> -->

### Captcha
Solve captacha images using CLIP and Object detection models.

```python
from clipcrop import clipcrop
# second arguement is the text prompt eg:image of cars
clipc = clipcrop.ClipCrop(image_path, "image of cars")
#loading models, processors, feature extractors
DFE, DM, CLIPM, CLIPP = clipc.load_models()
#generally keep high threshold to avoid noises
result = clipc.captcha(CLIPM, CLIPP, 4)
```

## Clip Segmentation

Segment out images using Detr Panoptic segmentation pipeline and leverage CLIP models to derive at the most probable one for your query

### Implementation

```python
from clipcrop import clipcrop
clipseg = clipcrop.ClipSeg("/content/input.png", "black colored car")
segmentor, clipmodel, clipprocessor = clipseg.load_models()
result = clipseg.segment_image(segmentor, clipmodel, clipprocessor)
# gives a list of dicitonary of top images and its relative similarity score and you can override this by setting num = 5  to get top 5 etc
```

