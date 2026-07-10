# pbl

Project-Based Learning (PBL) coursework: a deep learning-based image encryption system.

## Contents

| File | Description |
|---|---|
| `DL_Image_Encryption_System.ipynb` | A DCGAN-based encryption-key generator paired with an attention/residual CNN encryption-decryption network, plus a security-analysis module (histograms, pixel correlation, information entropy, NPCR/UACI) for evaluating the encryption scheme. Built with TensorFlow/Keras. |
| `PBL - I Review 2.docx` | Project review document/report submitted for the course. |
| `test/` | Scratch scripts used while developing the project. |

## How to run

This notebook was built for Google Colab with GPU acceleration. To run it:

```bash
git clone https://github.com/bharat3645/pbl.git
cd pbl
pip install tensorflow numpy matplotlib opencv-python scikit-learn scikit-image scipy tqdm
jupyter notebook DL_Image_Encryption_System.ipynb
```
