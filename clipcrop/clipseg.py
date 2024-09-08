import cv2
import numpy as np
from PIL import Image
from .components import ClipSegmentor


class ClipSeg:
    segmentor = ClipSegmentor("openai/clip-vit-base-patch32").load_models()[0]
    model = ClipSegmentor("openai/clip-vit-base-patch32").load_models()[1]
    processor = ClipSegmentor("openai/clip-vit-base-patch32").load_models()[2]

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

    def segment_image(self, num=None):

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

        segments = self.segmentor(self.input_path)
        img = Image.open(self.input_path)
        images_list = [self.unmask(img, x['mask']) for x in segments]
        inputs = self.processor(text=[self.input_text],
                                images=images_list,
                                return_tensors="pt",
                                padding=True)
        outputs = self.model(**inputs)
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
        except Exception as e:
            raise ValueError(e)
