import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, DetrFeatureExtractor, DetrForObjectDetection, pipeline
import torch
import cv2

class ClipCrop():

  def __init__(self, image_path, text, num=3):

    self.image_path = image_path
    self.text = text
    self.num = num

  def extract_image(self):

    image = Image.open(self.image_path)
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
    dmodel = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = dmodel(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    probas = outputs.logits.softmax(-1)[0, :, :-1] #removing no class as detr maps 

    keep = probas.max(-1).values > 0.96
    outs = feature_extractor.post_process(outputs, torch.tensor(image.size[::-1]).unsqueeze(0))
    bboxes_scaled = outs[0]['boxes'][keep].detach().numpy()
    labels = outs[0]['labels'][keep].detach().numpy()
    scores = outs[0]['scores'][keep].detach().numpy()

    images_list = []
    for i,j in enumerate(bboxes_scaled):
      
      xmin = int(j[0])
      ymin = int(j[1])
      xmax = int(j[2])
      ymax = int(j[3])

      im_arr = np.array(image)
      roi = im_arr[ymin:ymax, xmin:xmax]
      roi_im = Image.fromarray(roi)

      images_list.append(roi_im)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(text = [self.text], images=images_list , return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_text
    probs = logits_per_image.softmax(-1)
    l_idx = np.argsort(probs[-1].detach().numpy())[::-1][0:self.num]
    
    final_ims = []
    for i,j in enumerate(images_list):
      json_dict = {}
      if i in l_idx:
        json_dict['image'] = images_list[i]
        json_dict['score'] = probs[-1].detach().numpy()[i]

        final_ims.append(json_dict)

    fi = sorted(final_ims, key=lambda item: item.get("score"), reverse=True)
    return fi


class ClipSeg():
  def __init__(self, input_path, input_text):
    
    self.input_path = input_path
    self.input_text = input_text

  def load_models(self):
    
    segmentor = pipeline("image-segmentation")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") 

    return segmentor, model, processor

  def unmask(self, img, segim):

    self.img = img
    self.segim = segim

    mask = cv2.bitwise_not(np.array(self.segim))
    imask = cv2.bitwise_and(np.array(self.img), np.array(self.img), mask = np.array(self.segim)) 
    pil_mask = Image.fromarray(imask)

    return pil_mask
  
  def segment_image(self, segmentor, model, processor):
    
    self.segmentor = segmentor 
    self.model = model 
    self.processor = processor

    print("Extracting Segments")
    segments = self.segmentor(self.input_path)
    img = Image.open(self.input_path)
    images_list = [self.unmask(img, x['mask']) for x in segments]
    scores = [x['score'] for x in segments]

    print("Processing using CLIP to derive the most probable")
    inputs = processor(text = [self.input_text], images=images_list , return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_text
    probs = logits_per_image.softmax(-1).detach().numpy()
    most_prob = np.argmax(probs)

    print("Results saved to variables")
    return images_list[most_prob], scores[most_prob], self.input_text