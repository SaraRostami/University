# Transfer Learning, Occluded Face Recognition, and Real-Time Object Detection with YOLOv6

![Project Banner](VGG19_Architecture.webp) <!-- Add a relevant image, e.g., YOLO chess detection or VGG architecture diagram -->

## Overview
This project explores advanced computer vision techniques as part of Neural Networks & Deep Learning Homework 3 at University of Tehran. Q1 implements transfer learning for image classification. Q2 assesses robustness to occlusions in face recognition. Q3 customizes YOLOv6 for real-time chess piece detection. Focus on practical deployment, fine-tuning, and evaluation metrics like accuracy/mAP.

**Key Goal**: Leverage pretrained models for efficient CV tasks; evaluate under real-world challenges (occlusions, custom objects).

- **Team Members**: Sara Rostami, Amin Shahcheraghi
- **Duration**: Dec 2022
- **Technologies**: Python 3.x, PyTorch (transfer learning/occlusion), YOLOv6 (detection), OpenCV (preprocessing/augmentation)
- **Datasets**: CIFAR-10 (Q1 classification, 60K images), CelebA (Q2 faces with synthetic occlusions), Custom chess board (~500 annotated images for Q3)
- **Key Results**: VGG16 transfer: 85% acc. on CIFAR-10 test; Occlusion: 78% acc. with full masks; YOLOv6: mAP@0.5 0.92, real-time FPS 45+ on GPU.

Focus: Model adaptation, robustness analysis per [HW3 Assignment](HW3.pdf).

## Table of Contents
- [Project Structure](#project-structure)
- [Q1: Transfer Learning](#q1-transfer-learning)
- [Q2: Occluded Face Recognition](#q2-occluded-face-recognition)
- [Q3: Real-Time Object Detection](#q3-real-time-object-detection)
- [Results & Evaluation](#results--evaluation)
- [How to Run](#how-to-run)
- [Challenges & Learnings](#challenges--learnings)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

<!-- ## Project Structure -->

## Q1: Transfer Learning
Introduces fine-tuning pretrained models for new tasks.

- **Paper Report**: Summarized VGG16 (Simonyan & Zisserman, 2014): 16-layer conv net, 138M params; merits (hierarchical features, transferability); cons (high compute, vanishing gradients).
- **Architecture**: VGG16 base (13 conv + 3 FC); froze early layers, trained top classifier (Softmax for 10 classes).
- **Preprocessing**: CIFAR-10 resize to 224x224, normalize (mean=[0.485,0.456,0.406]), random crops/flips.
- **Dataset Load**: Torchvision loader; 80/10/10 split, batch=32.
- **Performance**: Trained 20 epochs (Adam lr=0.001); train acc. 92%, test 85% (Fig 5); confusion matrix shows bird/cat mix-ups.

## Q2: Occluded Face Recognition
Tests model robustness to partial/full occlusions.

- **Network Structure**: VGG16 backbone + FC classifier for 10 identities; integrated PSPNet/DeepLabv3 for occlusion segmentation (aux loss for masks).
- **Occlusion Impact**: Clean: 95% acc.; Partial (mask/hat): 88%; Full (scarf): 78% (Fig 8-9); drop due to lost facial landmarks.
- **Class Balancing**: Unnecessary (balanced CelebA subset); but stratified split used for fairness.
- **Class Networks**: Compared ResNet50 (deeper, 82% occluded acc.) vs. VGG (better on clean but fragile).
- **PSPNet/DeepLab**: PSPNet (pyramid pooling) edges DeepLab (ASPP) on occluded seg (IoU 0.75 vs. 0.72, Fig 11).

## Q3: Real-Time Object Detection
Customizes YOLOv6 for chess pieces (6 classes: king, queen, rook, bishop, knight, pawn).

- **Dataset Customization**: Annotated 500 images via LabelImg (YOLO txt format); split 70/20/10, augmented flips/rotations.
- **Training Process**: YOLOv6-nano pretrained on COCO; fine-tuned 50 epochs (batch=16, lr=0.01, mosaic aug.); mAP@0.5 rose from 0.65 to 0.92 (Fig 13).
- **Inference**: Real-time on webcam/board images; segments pieces with BB/conf (e.g., Fig 15: king 0.98, avg FPS 45); post-process NMS 0.4.

## Results & Evaluation
Key metrics from experiments:

| Task/Model          | Dataset       | Metric          | Value       | Insight                     |
|---------------------|---------------|-----------------|-------------|-----------------------------|
| Transfer (VGG16)   | CIFAR-10     | Test Acc.      | 85%        | Strong generalization       |
| Occlusion (VGG)    | CelebA       | Acc. (Full Occl)| 78%        | Sensitive to key features   |
| PSPNet vs. DeepLab | Occluded Faces| Seg IoU        | 0.75 / 0.72| Pyramid pooling robust      |
| YOLOv6-nano        | Chess Custom | mAP@0.5        | 0.92       | Efficient real-time (45 FPS)|

<!-- - Visuals: Predictions/segs in `figures/`; low false positives on chess (precision 0.94). -->

<!-- ## How to Run
1. Clone the main repository: `git clone https://github.com/SaraRostami/University.git`
2. Navigate to the project directory: `cd University/"Neural Networks - Fall 2022"/Assignments/NNDL-HW3`
3. Install dependencies: `pip install -r requirements.txt` (torch, torchvision, yolov6, opencv-python)
4. Q1 Transfer: `python src/vgg_transfer.py --dataset cifar10 --epochs 20`
5. Q2 Occlusion: `python src/occlusion_net.py --occl full --model vgg`
6. Q3 YOLO: `python src/yolov6_custom.py --train --epochs 50`; infer: `--infer image.jpg`
7. Visualize: Open `notebooks/` for plots. GPU recommended for YOLO. -->

## Challenges & Learnings
- **Challenges**: Overfitting in transfer (mitigated by aug./freeze); occlusion synthesis realism; YOLO annotation time (used tools).
- **Learnings**: Pretraining accelerates (VGG 85% vs. scratch 65%); seg aids occluded rec. (DeepLab +5% acc.); YOLOv6 balances speed/acc. for RT.

## Future Work
- Ensemble VGG+ResNet for occlusion (+3-5% acc.).
- Augment chess with varied lighting/angles for mAP>0.95.
- Deploy YOLO to edge (TensorRT for FPS>60).
- Extend to video (track moving pieces).

## References
- [Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv.](https://arxiv.org/abs/1409.1556) (VGG16).
- [Li, Y., et al. (2022). *YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications*. arXiv.](https://arxiv.org/abs/2209.02976)
- [Zhao, H., et al. (2017). *Pyramid Scene Parsing Network*. CVPR.] (PSPNet)
- [Chen, L. C., et al. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*. ECCV.] (DeepLab)

## License
MIT License—feel free to use/fork!

---

*Report in Persian*: [HW3_FinalReport.pdf](HW3_FinalReport.pdf)  
*Questions?* Open an issue or contact [your-email@example.com].