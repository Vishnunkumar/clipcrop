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
  
  def load_models(self):

    DFE = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
    DM = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
    CLIPM = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPP = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return DFE, DM, CLIPM, CLIPP

  def extract_image(self, DFE, DM, CLIPM, CLIPP):

    self.DFE = DFE
    self.DM = DM
    self.CLIPM = CLIPM
    self.CLIPP = CLIPP

    image = Image.open(self.image_path)

    inputs = self.DFE(images=image, return_tensors="pt")
    outputs = self.DM(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    probas = outputs.logits.softmax(-1)[0, :, :-1] #removing no class as detr maps 

    keep = probas.max(-1).values > 0.96
    outs = self.DFE.post_process(outputs, torch.tensor(image.size[::-1]).unsqueeze(0))
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

    inputs = self.CLIPP(text = [self.text], images=images_list , return_tensors="pt", padding=True)
    outputs = self.CLIPM(**inputs)
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

  # def bwimage(self, img, segim):
    
  #   self.img = img
  #   self.segim = segim

  #   gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
  #   imask = cv2.bitwise_and(np.array(gray), np.array(gray), mask = np.array(self.segim))
  #   pil_mask = Image.fromarray(imask)

  #   return pil_mask


  def segment_image(self, segmentor, model, processor, num=None):
    
    self.segmentor = segmentor 
    self.model = model 
    self.processor = processor
    self.num = num

    segments = self.segmentor(self.input_path)
    img = Image.open(self.input_path)
    images_list = [self.unmask(img, x['mask']) for x in segments]
    scores = [x['score'] for x in segments]
    inputs = self.processor(text = [self.input_text], images=images_list , return_tensors="pt", padding=True)
    outputs = self.model(**inputs)
    logits_per_image = outputs.logits_per_text
    probs = logits_per_image.softmax(-1).detach().numpy()
    res_list = np.argsort(probs[0])[::-1]

    if self.num is None:
      self.num = 1

    seg_list = []
    for x in res_list[:self.num]:
      seg_dict = {}
      seg_dict["image"] = images_list[x]
      seg_dict["score"] = scores[x]
      seg_list.append(seg_dict)
    
    print(seg_list)
    res_im = seg_list[0]['image']
    res_cv = np.array(res_im)
    alpha = np.sum(res_cv, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    res = np.dstack((res_cv, alpha))
    respl = Image.fromarray(res)
    
    return respl, seg_list[0]['score']