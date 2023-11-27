from non_cnn.detection import get_sub_images
from non_cnn.model import FFTModel
from cnn.cnn import predict_and_annotate, convert_to_pil_image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
from PIL import Image

m_col = {
    'car': (0, 0, 255),
    'bus': (0, 255, 0),
    'truck': (255, 0, 0),
}


def yolo_predict(model, img):
    res = predict_and_annotate(model, img)
    return convert_to_pil_image(res)


def multi_predict(model, img):
    images, bbox = get_sub_images(img)
    ann = Annotator(img)
    for i, im in enumerate(images):
        p = model.single_predict(Image.fromarray(im).resize((256, 256)))
        first_pred = list(p.keys())[0]
        if first_pred == 'none':
            continue
        ann.box_label(bbox[i], label=f"{first_pred} {p[first_pred]:.2f}", color=m_col[first_pred])
    return cv2.cvtColor(ann.im, cv2.COLOR_BGR2RGB)


MODELS = {
    'SVC-Single': {
        'type': 'non_cnn',
        'path': 'modelSVC4.pkl',
        'video': False,
        'multilabel': False,
        'loader': FFTModel.load,
        'predict': lambda m, img: m.single_predict(img),
    },
    'SVC-Multi': {
        'from': 'SVC-Single',
        'video': False,
        'multilabel': True,
        'predict': multi_predict,
        'img_loader': lambda img: cv2.imread(img),
    },
    'CNN-medium': {
        'type': 'cnn',
        'path': 'medium vehicle.pt',
        'video': True,
        'multilabel': True,
        'loader': YOLO,
        'predict': yolo_predict,
    },
    'CNN-nano': {
        'type': 'cnn',
        'path': 'nano vehicle.pt',
        'video': True,
        'multilabel': True,
        'loader': YOLO,
        'predict': yolo_predict,
    }
}
