from mmdet.apis import init_detector, inference_detector

config = 'pytorch_D-RISE/mmdet_configs/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
checkpoint = 'faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
device = 'cuda:0'

model = init_detector(config, checkpoint, device)
label_names = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm

image = cv2.imread('pytorch_D-RISE/images/cake.jpg')
scale = 600 / min(image.shape[:2])
image = cv2.resize(image,
                   None,
                   fx=scale,
                   fy=scale,
                   interpolation=cv2.INTER_AREA)
plt.figure(figsize=(7, 7))
plt.imshow(image[:, :, ::-1])
plt.savefig('Image11.png')


out = inference_detector(model, image)
res = image.copy()
for i, pred in enumerate(out):
    for *box, score in pred:
        if score < 0.5:
            break
        box = tuple(np.round(box).astype(int).tolist())
        print(i, label_names[i], box, score)
        cv2.rectangle(res, box[:2], box[2:], (0, 255, 0), 5)

plt.figure(figsize=(7, 7))
plt.imshow(res[:, :, ::-1])
plt.savefig('Image21.png')

def generate_mask(image_size, grid_size, prob_thresh):
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

    mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
            prob_thresh).astype(np.float32)
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = np.random.randint(0, cell_w)
    offset_h = np.random.randint(0, cell_h)
    mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
    return mask

def mask_image(image, mask):
    masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
              255).astype(np.uint8)
    return masked

np.random.seed(0)

image_h, image_w = image.shape[:2]

images = []
for _ in range(25):
    mask = generate_mask(image_size=(image_w, image_h),
                         grid_size=(16, 16),
                         prob_thresh=0.5)
    masked = mask_image(image, mask)
    out = inference_detector(model, masked)
    res = masked.copy()
    for pred in out:
        for *box, score in pred:
            if score < 0.5:
                break
            box = tuple(np.round(box).astype(int).tolist())
            cv2.rectangle(res, box[:2], box[2:], (0, 255, 0), 5)
    images.append(res)

fig = plt.figure(figsize=(15, 10))
axes = fig.subplots(5, 5)
for i in range(5):
    for j in range(5):
        axes[i][j].imshow(images[i * 5 + j][:, :, ::-1])
        axes[i][j].axis('off')
plt.tight_layout()
plt.savefig('Image31.png')


def iou(box1, box2):
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
    br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
    intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
    area1 = np.prod(box1[2:] - box1[:2])
    area2 = np.prod(box2[2:] - box2[:2])
    return intersection / (area1 + area2 - intersection)

def generate_saliency_map(image,
                          target_class_index,
                          target_box,
                          prob_thresh=0.5,
                          grid_size=(16, 16),
                          n_masks=5000,
                          seed=0):
    np.random.seed(seed)
    image_h, image_w = image.shape[:2]
    res = np.zeros((image_h, image_w), dtype=np.float32)
    for _ in range(n_masks):
        mask = generate_mask(image_size=(image_w, image_h),
                             grid_size=grid_size,
                             prob_thresh=prob_thresh)
        masked = mask_image(image, mask)
        out = inference_detector(model, masked)
        pred = out[target_class_index]
        score = max([iou(target_box, box) * score for *box, score in pred],
                    default=0)
        res += mask * score
    return res

####################################################################################

target_box = np.array([147, 432, 262, 557])
saliency_map = generate_saliency_map(image,
                                     target_class_index=54,
                                     target_box=target_box,
                                     prob_thresh=0.5,
                                     grid_size=(16, 16),
                                     n_masks=1000)

image_with_bbox = image.copy()
cv2.rectangle(image_with_bbox, tuple(target_box[:2]), tuple(target_box[2:]),
              (0, 255, 0), 5)
plt.figure(figsize=(7, 7))
plt.imshow(image_with_bbox[:, :, ::-1])
plt.imshow(saliency_map, cmap='jet', alpha=0.5)
plt.axis('off')
plt.savefig('Image41.png')

####################################################################################

target_box = np.array([263, 440, 384, 562])
saliency_map = generate_saliency_map(image,
                                     target_class_index=54,
                                     target_box=target_box,
                                     prob_thresh=0.5,
                                     grid_size=(16, 16),
                                     n_masks=1000)

image_with_bbox = image.copy()
cv2.rectangle(image_with_bbox, tuple(target_box[:2]), tuple(target_box[2:]),
              (0, 255, 0), 5)
plt.figure(figsize=(7, 7))
plt.imshow(image_with_bbox[:, :, ::-1])
plt.imshow(saliency_map, cmap='jet', alpha=0.5)
plt.axis('off')
plt.savefig('Image51.png')

####################################################################################

target_box = np.array([372, 427, 478, 544])
saliency_map = generate_saliency_map(image,
                                     target_class_index=54,
                                     target_box=target_box,
                                     prob_thresh=0.5,
                                     grid_size=(16, 16),
                                     n_masks=1000)

image_with_bbox = image.copy()
cv2.rectangle(image_with_bbox, tuple(target_box[:2]), tuple(target_box[2:]),
              (0, 255, 0), 5)
plt.figure(figsize=(7, 7))
plt.imshow(image_with_bbox[:, :, ::-1])
plt.imshow(saliency_map, cmap='jet', alpha=0.5)
plt.axis('off')
plt.savefig('Image61.png')

####################################################################################

target_box = np.array([125, 226, 479, 480])
saliency_map = generate_saliency_map(image,
                                     target_class_index=54,
                                     target_box=target_box,
                                     prob_thresh=0.5,
                                     grid_size=(16, 16),
                                     n_masks=1000)

image_with_bbox = image.copy()
cv2.rectangle(image_with_bbox, tuple(target_box[:2]), tuple(target_box[2:]),
              (0, 255, 0), 5)
plt.figure(figsize=(7, 7))
plt.imshow(image_with_bbox[:, :, ::-1])
plt.imshow(saliency_map, cmap='jet', alpha=0.5)
plt.axis('off')
plt.savefig('Image71.png')

####################################################################################

target_box = np.array([83, 227, 521, 608])
saliency_map = generate_saliency_map(image,
                                     target_class_index=55,
                                     target_box=target_box,
                                     prob_thresh=0.5,
                                     grid_size=(16, 16),
                                     n_masks=1000)

image_with_bbox = image.copy()
cv2.rectangle(image_with_bbox, tuple(target_box[:2]), tuple(target_box[2:]),
              (0, 255, 0), 5)
plt.figure(figsize=(7, 7))
plt.imshow(image_with_bbox[:, :, ::-1])
plt.imshow(saliency_map, cmap='jet', alpha=0.5)
plt.axis('off')
plt.savefig('Image81.png')

####################################################################################

target_box = np.array([0, 306, 600, 845])
saliency_map = generate_saliency_map(image,
                                     target_class_index=60,
                                     target_box=target_box,
                                     prob_thresh=0.5,
                                     grid_size=(16, 16),
                                     n_masks=1000)

image_with_bbox = image.copy()
cv2.rectangle(image_with_bbox, tuple(target_box[:2]), tuple(target_box[2:]),
              (0, 255, 0), 5)
plt.figure(figsize=(7, 7))
plt.imshow(image_with_bbox[:, :, ::-1])
plt.imshow(saliency_map, cmap='jet', alpha=0.5)
plt.axis('off')
plt.savefig('Image91.png')




