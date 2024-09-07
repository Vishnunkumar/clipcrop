import cv2
import numpy as np
import pytesseract
import torch
from PIL import Image
from .components import ClipCropper, ClipSegmentor


# The ClipCrop class is used for cropping and clipping images.
class ClipCrop:

    def __init__(self, image_path):

        self.image_path = image_path

    def auto_captcha(self,
                     clipm=ClipCropper("openai/clip-vit-base-patch32").load_models()[2],
                     clipp=ClipCropper("openai/clip-vit-base-patch32").load_models()[3],
                     threshold=3,
                     tx="default"):

        self.threshold = threshold
        self.clipm = clipm
        self.clipp = clipp
        self.tx = tx

        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get the text from the image
        ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        close = cv2.erode(thresh, kernel, iterations=2)
        invert = cv2.bitwise_not(close)
        inf = pytesseract.image_to_string(invert)

        if len(inf.split('\n')) > 1:
            txt = [x for x in inf.split('\n') if len(x) > 2][1]
        else:
            txt = self.tx

        # Process for captcha resolution
        blur = cv2.medianBlur(gray, 3)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

        # Threshold and morph close
        thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours and filter using threshold area
        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        image_number = 0
        img_list = []
        r_list = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w / h == 1 and (w, h) > (5, 5):
                ROI = image[y:y + h, x:x + w]
                img_list.append(Image.fromarray(ROI[:, :, ::-1]))
                r_list.append([(x, y), (x + w, y + h)])
                image_number += 1

        inps = self.clipp(text=["images with " + txt], images=img_list, return_tensors="pt", padding=True)
        outs = self.clipm(**inps)
        logits_per_image = outs.logits_per_text
        probs = logits_per_image.softmax(-1)
        l_idx = list(np.argsort(probs[-1].detach().numpy())[::-1][0:self.threshold])

        for x in l_idx:
            cv2.rectangle(image, r_list[x][0], r_list[x][1], (36, 255, 12), 2)

        return Image.fromarray(image[:, :, ::-1])

    def extract_image(self,
                      text,
                      det_processor=ClipCropper("openai/clip-vit-base-patch32").load_models()[0],
                      det_model=ClipCropper("openai/clip-vit-base-patch32").load_models()[1],
                      clipm=ClipCropper("openai/clip-vit-base-patch32").load_models()[2],
                      clipp=ClipCropper("openai/clip-vit-base-patch32").load_models()[3],
                      num=3):

        self.det_processor = det_processor
        self.det_model = det_model
        self.clipm = clipm
        self.clipp = clipp
        self.num = num
        self.text = text

        image = cv2.imread(self.image_path)
        inputs = self.det_processor(images=image, return_tensors="pt")
        outputs = self.det_model(**inputs)

        # model predicts bounding boxes and corresponding COCO classes
        probabilities = outputs.logits.softmax(-1)[0, :, :-1]  # removing no class as detr maps

        keep = probabilities.max(-1).values > 0.6
        outs = self.det_processor.post_process(outputs, torch.tensor(image.shape[:2]).unsqueeze(0))
        bboxes_scaled = outs[0]['boxes'][keep].detach().numpy()

        images_list = []
        for i, j in enumerate(bboxes_scaled):
            xmin = int(j[0])
            ymin = int(j[1])
            xmax = int(j[2])
            ymax = int(j[3])

            roi = image[ymin:ymax, xmin:xmax]
            roi_im = Image.fromarray(roi[:, :, ::-1])
            images_list.append(roi_im)

        inputs = self.clipp(text=[self.text],
                            images=images_list,
                            return_tensors="pt",
                            padding=True)
        outputs = self.clipm(**inputs)
        logits_per_image = outputs.logits_per_text
        probs = logits_per_image.softmax(-1)
        l_idx = np.argsort(probs[-1].detach().numpy())[::-1][0:self.num]

        final_ims = []
        for i, j in enumerate(images_list):
            json_dict = {}
            if i in l_idx:
                json_dict['image'] = images_list[i]
                json_dict['score'] = probs[-1].detach().numpy()[i]

                final_ims.append(json_dict)

        fi = sorted(final_ims, key=lambda item: item.get("score"), reverse=True)
        return fi


# The ClipSeg class is a placeholder for a more detailed implementation.
class ClipSeg:

    def __init__(self, input_path, input_text):

        self.input_path = input_path
        self.input_text = input_text

    def unmask(self, img, segim):

        self.img = img
        self.segim = segim

        mask = cv2.bitwise_not(np.array(self.segim))
        imask = cv2.bitwise_and(np.array(self.img), np.array(self.img), mask=np.array(self.segim))
        pil_mask = Image.fromarray(imask)

        return pil_mask

    def segment_image(self,
                      segmentor=ClipSegmentor("openai/clip-vit-base-patch32").load_models()[0],
                      model=ClipSegmentor("openai/clip-vit-base-patch32").load_models()[1],
                      processor=ClipSegmentor("openai/clip-vit-base-patch32").load_models()[2],
                      num=None):

        self.segmentor = segmentor
        self.model = model
        self.processor = processor
        self.num = num

        segments = self.segmentor(self.input_path)
        img = Image.open(self.input_path)
        images_list = [self.unmask(img, x['mask']) for x in segments]
        scores = [x['score'] for x in segments]
        inputs = self.processor(text=[self.input_text], images=images_list, return_tensors="pt", padding=True)
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
            resp = Image.fromarray(res)
            seg_dict["image"] = resp
            seg_dict["score"] = scores[x]
            seg_list.append(seg_dict)

        return seg_list

    def remove_background(self):

        segmentor = ClipSegmentor("openai/clip-vit-base-patch32").load_models()[0]
        model = ClipSegmentor("openai/clip-vit-base-patch32").load_models()[1]
        processor = ClipSegmentor("openai/clip-vit-base-patch32").load_models()[2]

        segments = segmentor(self.input_path)
        img = Image.open(self.input_path)
        images_list = [self.unmask(img, x['mask']) for x in segments]
        inputs = processor(text=[self.input_text], images=images_list, return_tensors="pt", padding=True)
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
            response_image = Image.fromarray(res)

        try:
            return response_image
        except:
            raise ValueError("response image is null")
