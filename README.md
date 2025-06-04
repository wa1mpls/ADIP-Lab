#  Advanced Computer Vision Projects 

##  Project List

### 1. Morphological Operations on Grayscale Images
**Keywords**: Dilation, Erosion, Opening, Closing, Top-Hat, Gradient, Granulometry, Reconstruction

- Implemented grayscale morphological operators **manually from scratch** and also using OpenCV.
- Modular code in `morphological_operations/grayscale.py`.
- Includes visualization, performance comparison, and granulometry analysis.
- Granulometry values are output numerically and plotted.
- Command-line usage via `main_grayscale.py`.

Try:
```bash
python main_grayscale.py -i input.jpg -o output.jpg -p "Dilate" -m manual
```

---

### 2. Vision Transformer (ViT) – Built from Scratch
**Keywords**: Transformer Encoder, Attention, MLP Head, CIFAR-10, Training Pipeline

- Full Vision Transformer implemented from scratch using PyTorch.
- Dataset: CIFAR-10 (50,000 training / 10,000 test)
- Includes:
  - Multi-head Attention
  - Transformer Block
  - CLS token classification head
  - Experiment logging, loss visualization, accuracy comparison
- All experiments performed yielded **Top-1 Accuracy = 10%**, indicating model instability or misconfiguration.

 Key directories:
- `models/`: ViT architecture modules
- `training/`, `evaluation/`, `utils/`, `logs/`
- `main.py`: entry point

---

### 3. Generative Adversarial Network (GAN) – Handwritten Digits
**Keywords**: Generator, Discriminator, Binary Cross Entropy, MNIST, GAN instability

- GAN implemented and trained using **PyTorch** on **MNIST** dataset.
- Generator maps noise to synthetic digits, Discriminator distinguishes real/fake.
- Trained for 50 and 100 epochs.
- Observed **training instability** and **mode collapse**; output images were blurry or repetitive.
- Suggested fixes: label smoothing, feature matching, spectral normalization, longer training.

➡ Highlights:
- Defined custom `Generator` and `Discriminator` using `nn.Linear`
- Manual training loop with separate optimizer and loss logic
- Plotted generated outputs and loss trends for each epoch

---

##  Technologies Used

- Python 3.x
- PyTorch
- NumPy, Matplotlib
- OpenCV
- Google Colab (for training on GPU)

---

## ▶ How to Run

Each project includes a `source/` folder with runnable code. For example:

```bash
cd morphological_operations
python main_grayscale.py -i lena.png -o dilated.png -p "Dilate" -m opencv
```

For ViT:

```bash
python main.py
```

---

##  Summary

| Project           | Dataset     | Main Task                        | Outcome/Challenge                 |
|------------------|-------------|----------------------------------|-----------------------------------|
| Morphology       | Any image   | Grayscale operations             | Verified vs OpenCV                |
| Vision Transformer | CIFAR-10  | Image classification             | Low accuracy, needs tuning        |
| GAN              | MNIST       | Image generation                 | Blurry outputs, unstable training |

---

## Author
GitHub: [wa1mpls](https://github.com/wa1mpls)

---

## Notes

All code was developed and tested in academic environments. For deployment or reproduction, ensure dependencies are installed and GPU acceleration is available for ViT and GAN projects.
