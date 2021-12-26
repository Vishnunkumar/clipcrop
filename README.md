# clipcrop
Extract sections of images from your image by using OpenAI's CLIP and Facebooks Detr implemented on HuggingFace Transformers

## Implementation

```python
clipcrop = ClipCrop("/content/nm.jpg", "black car")
l_images = clipcrop.extract_image()

# gives a list of dicitonary of top3 images and its relative similarity score and you can override this by setting num = 5  to get top 5 etc
```



