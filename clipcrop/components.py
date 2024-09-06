from transformers import CLIPProcessor, CLIPModel, pipeline, YolosImageProcessor, YolosForObjectDetection


class LoadClipModels:
    def load_models(self):
        pass


class ClipCropper(LoadClipModels):
    def __init__(self, openai):
        self.openai = openai

    def load_models(self):
        det_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")
        det_detector = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')
        clip_model = CLIPModel.from_pretrained(self.openai)
        clip_processor = CLIPProcessor.from_pretrained(self.openai)

        return det_processor, det_detector, clip_model, clip_processor


class ClipSegmentor(LoadClipModels):
    def __init__(self, openai):
        self.openai = openai

    def load_models(self):
        segmentor = pipeline("image-segmentation",
                             model="facebook/maskformer-swin-tiny-coco")
        model = CLIPModel.from_pretrained(self.openai)
        processor = CLIPProcessor.from_pretrained(self.openai)

        return segmentor, model, processor
