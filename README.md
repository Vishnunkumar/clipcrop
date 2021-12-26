# clipcrop
Extract sections of images from your image by using OpenAI's CLIP and Facebooks Detr implemented on HuggingFace Transformers

## Implementation

```python
clipcrop = ClipCrop("/content/nm.jpg", "woman in white frock")
lm = clipcrop.extract_image()

# gives a list of dicitonary of top3 images and its relative similarity score and you can override this by setting num = 5  to get top 5 etc
```

## Result

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
</p>

- Inspired from [@vijishmadhavan](https://github.com/vijishmadhavan/Crop-CLIP/)

