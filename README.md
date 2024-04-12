# clipcrop
- Extract sections of images from your image by using OpenAI's CLIP and YoloSmall implemented on HuggingFace Transformers
- Added new capability for segmentation using CLIP and Detr segmentation models

# Installation
```python
pip install clipcrop
```

## Clip Crop

Extract sections of images from your image by using OpenAI's CLIP and YoloSmall implemented on HuggingFace Transformers 

### Extraction

```python
from clipcrop import clipcrop

cc = clipcrop.ClipCrop("/content/sample.jpg")

DFE, DM, CLIPM, CLIPP = cc.load_models()

result = cc.extract_image(DFE, DM, CLIPM, CLIPP, "text content", num=2)

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
Solve captacha images using CLIP and Object detection models. Ensure Tesseract is installed and executable in your path

```python
from clipcrop import clipcrop

cc = clipcrop.ClipCrop(image_path)

DFE, DM, CLIPM, CLIPP = cc.load_models()

result = cc.auto_captcha(CLIPM, CLIPP, 4)

```

## Clip Segmentation

Segment out images using Detr Panoptic segmentation pipeline and leverage CLIP models to derive the most probable one for your query

### Extraction

```python

from clipcrop import clipcrop

clipseg = clipcrop.ClipSeg("/content/input.png", "black colored car")

segmentor, clipmodel, clipprocessor = clipseg.load_models()

result = clipseg.segment_image(segmentor, clipmodel, clipprocessor)

```

### Remove Background
```python

from clipcrop import clipcrop

clipseg = clipcrop.ClipSeg("/content/input.png", "black colored car")

result = clipseg.remove_background()

```

### Other projects
- [SnapCode : Extract code blocks from images mixed with normal text](https://github.com/Vishnunkumar/snapcode)
- [HuggingFaceInference: Inference of different uses cases of finetued models](https://github.com/Vishnunkumar/huggingfaceinference)

### Contact
- Feel free to contact me on "nkumarvishnu25@gmail.com"
