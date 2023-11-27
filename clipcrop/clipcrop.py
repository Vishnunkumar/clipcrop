import numpy as np
import torch 
import cv2
import pytesseract
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, pipeline, YolosImageProcessor, YolosForObjectDetection

# The ClipCrop class is used for cropping and clipping images.
class ClipCrop():

  def __init__(self, image_path):

    self.image_path = image_path
    
  def load_models(self):
    """
    The function "load_models" loads pre-trained models for object detection and image-text matching.
    :return four objects: DFE (DetectionFeatureExtractor), DM (DetectionModelObjectDetection), CLIPM (CLIPModel),
    and CLIPP (CLIPProcessor).
    """

    DFE = YolosImageProcessor.from_pretrained("hustvl/yolos-small")
    DM = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')
    CLIPM = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPP = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return DFE, DM, CLIPM, CLIPP

  def auto_captcha(self, CLIPM, CLIPP, th=3, tx ="default"):
    """
    The `auto_captcha` function takes an image, extracts the text from it, processes it to resolve the
    captcha, and returns the image with highlighted regions where the captcha is detected.
    
    :param CLIPM: CLIPM is an instance of the CLIP model. It is used for performing inference on the
    images
    :param CLIPP: CLIPP is an instance of the CLIPProcessor class, which is used for processing text
    and images for the CLIP model. It is used to preprocess the text and images before passing them to
    the CLIP model for inference
    :param th: The parameter "th" stands for threshold and it determines the number of top predictions
    to consider for drawing rectangles on the image, defaults to 3 (optional)
    :param tx: The parameter `tx` is a default text that will be used if the captcha text cannot be
    extracted from the image, defaults to default (optional)
    :return: an image with rectangles drawn around the identified captcha regions.
    """
    
    self.th = th
    self.CLIPM = CLIPM
    self.CLIPP = CLIPP
    self.tx = tx

    image = cv2.imread(self.image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the text from the image
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    close = cv2.erode(thresh, kernel, iterations=2)
    invert = cv2.bitwise_not(close)
    inf = pytesseract.image_to_string(invert)
    
    if len(inf.split('\n')) > 1:
      txt = [x for x in inf.split('\n') if len(x) > 2][1]
    else:
      txt = self.tx

    # Process for captcha resolution
    blur = cv2.medianBlur(gray, 3)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    # Threshold and morph close
    thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours and filter using threshold area
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    image_number = 0
    img_list = []
    r_list = []
    for c in cnts:    
        x,y,w,h = cv2.boundingRect(c)
        if w/h == 1 and (w, h) > (5, 5):
            ROI = image[y:y+h, x:x+w]
            img_list.append(Image.fromarray(ROI[:,:,::-1]))
            r_list.append([(x, y), (x + w, y + h)])
            image_number += 1

    inps = self.CLIPP(text = ["images with " + txt], images=img_list , return_tensors="pt", padding=True)
    outs = self.CLIPM(**inps)
    logits_per_image = outs.logits_per_text
    probs = logits_per_image.softmax(-1)
    l_idx = list(np.argsort(probs[-1].detach().numpy())[::-1][0:self.th])

    for x in l_idx:
      cv2.rectangle(image, r_list[x][0], r_list[x][1], (36,255,12), 2)

    return Image.fromarray(image[:,:,::-1])

  def extract_image(self, DFE, DM, CLIPM, CLIPP, text, num=3):
    """
    The function `extract_image` takes an image, applies object detection using the DFE and DM models,
    extracts the bounding boxes and labels of the detected objects, crops the image based on the
    bounding boxes, and then uses the CLIPM model to classify the cropped images based on the provided
    text. The function returns a list of the top `num` cropped images along with their corresponding
    scores.
    
    :param DFE: DFE stands for "Detection and Feature Extraction" model. It is used to detect objects
    in an image and extract their features
    :param DM: The parameter `DM` is a model that predicts bounding boxes and corresponding COCO
    classes in an image
    :param CLIPM: CLIPM is an instance of the CLIP model, which is used for text-to-image retrieval.
    It takes in text and images as input and produces logits per text, indicating the similarity
    between the text and each image
    :param CLIPP: CLIPP is an instance of the CLIPProcessor class, which is used to preprocess text
    and images for input to the CLIP model. It handles tasks such as tokenization, encoding, and
    padding
    :param text: The "text" parameter is a string that represents the input text for the CLIP model.
    It is used to generate image-text embeddings and compare them to find the most relevant images
    :param num: The `num` parameter specifies the number of images to be extracted and returned as
    output, defaults to 3 (optional)
    :return: a list of dictionaries, where each dictionary contains an image and its corresponding
    score. The images are extracted from the input image based on predicted bounding boxes, and the
    scores are calculated using a CLIP model. The list is sorted in descending order based on the
    scores.
    """

    self.DFE = DFE
    self.DM = DM
    self.CLIPM = CLIPM
    self.CLIPP = CLIPP
    self.num = num
    self.text = text
    
    image = cv2.imread(self.image_path)
    inputs = self.DFE(images=image, return_tensors="pt")
    outputs = self.DM(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    probas = outputs.logits.softmax(-1)[0, :, :-1] #removing no class as detr maps 

    keep = probas.max(-1).values > 0.95
    outs = self.DFE.post_process(outputs, torch.tensor(image.shape[:2]).unsqueeze(0))
    bboxes_scaled = outs[0]['boxes'][keep].detach().numpy()
    labels = outs[0]['labels'][keep].detach().numpy()
    scores = outs[0]['scores'][keep].detach().numpy()

    images_list = []
    for i,j in enumerate(bboxes_scaled):
      
      xmin = int(j[0])
      ymin = int(j[1])
      xmax = int(j[2])
      ymax = int(j[3])

      roi = image[ymin:ymax, xmin:xmax]
      roi_im = Image.fromarray(roi[:,:,::-1])
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


# The ClipSeg class is a placeholder for a more detailed implementation.
class ClipSeg():
  
  def __init__(self, input_path, input_text):
    """
    The above function is a constructor that initializes the input_path and input_text attributes of
    an object.
    
    :param input_path: The input_path parameter is a string that represents the path to a file. It is
    used to specify the location of the file that contains the input text
    :param input_text: The `input_text` parameter is a string that represents the text input for your
    program. It could be any text that you want to process or manipulate in some way
    """
    
    self.input_path = input_path
    self.input_text = input_text

  def load_models(self):
    """
    The function "load_models" loads and returns three models: an image segmentation model, a CLIP
    model, and a CLIP processor.
    :return: three objects: `segmentor`, `model`, and `processor`.
    """
    
    segmentor = pipeline("image-segmentation", model="facebook/maskformer-swin-tiny-coco")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") 

    return segmentor, model, processor

  def unmask(self, img, segim):
    """
    The function takes an image and a segmentation mask as input, and returns the image with the masked
    regions unmasked.
    
    :param img: The `img` parameter is the input image that you want to unmask. It can be a numpy array
    or a PIL image
    :param segim: The `segim` parameter is a binary image that represents a segmentation mask. It is
    used to mask out certain regions of the input image (`img`). The regions that are masked out will
    be set to black (0) in the output image
    :return: a PIL (Python Imaging Library) image object.
    """

    self.img = img
    self.segim = segim

    mask = cv2.bitwise_not(np.array(self.segim))
    imask = cv2.bitwise_and(np.array(self.img), np.array(self.img), mask = np.array(self.segim)) 
    pil_mask = Image.fromarray(imask)

    return pil_mask

  def segment_image(self, segmentor, model, processor, num=None):
    """
    The `segment_image` function takes an input image, segments it using a given segmentor, processes
    the segments using a given model and processor, and returns a list of segmented images along with
    their scores.
    
    :param segmentor: The `segmentor` parameter is a function that takes an input image and returns a
    list of segments. Each segment is represented as a dictionary with keys "mask" and "score". The
    "mask" value is a binary mask indicating the pixels belonging to the segment, and the "score"
    value
    :param model: The `model` parameter refers to a machine learning model that is used for image
    segmentation. It takes in an image as input and outputs a segmented image, where different regions
    of the image are assigned different labels or classes
    :param processor: The `processor` parameter is an object that is used to preprocess the input text
    and images before feeding them into the model. It is responsible for tasks such as tokenization,
    padding, and converting the input into a format that the model can understand
    :param num: The `num` parameter specifies the number of segments to be returned. If `num` is not
    provided, it defaults to 1, meaning only the top-scoring segment will be returned
    :return: a list of dictionaries, where each dictionary contains an "image" key with the segmented
    image and a "score" key with the corresponding score for that segment.
    """
    
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
      res_im = images_list[x]
      res_cv = np.array(res_im)
      nz = np.sum(res_cv, axis=-1) > 0
      nz = np.uint8(nz * 255)
      res = np.dstack((res_cv, nz))
      respl = Image.fromarray(res)  
      seg_dict["image"] = respl
      seg_dict["score"] = scores[x]
      seg_list.append(seg_dict)
    
    return seg_list
  
  def remove_background(self):
    """
    The function `remove_background` takes an input image, segments it into different regions, and
    removes the background from the image using a trained model.
    :return: an Image object named `response_Image`.
    """

    segmentor = self.load_models()[0] 
    model = self.load_models()[1]
    processor = self.load_models()[2]

    segments = segmentor(self.input_path)
    img = Image.open(self.input_path)
    images_list = [self.unmask(img, x['mask']) for x in segments]
    scores = [x['score'] for x in segments]
    inputs = processor(text = [self.input_text], images=images_list , return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_text
    probs = logits_per_image.softmax(-1).detach().numpy()
    res_list = np.argsort(probs[0])[::-1]

    for x in res_list[:1]:
      res_im = images_list[x]
      res_cv = np.array(res_im)
      nz = np.sum(res_cv, axis=-1) > 0
      nz = np.uint8(nz * 255)
      res = np.dstack((res_cv, nz))
      response_Image = Image.fromarray(res)  

    return response_Image
