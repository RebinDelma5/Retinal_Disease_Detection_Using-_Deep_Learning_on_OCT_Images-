# Retinal Disease Detection Using Deep Learning on OCT Images

Classifies retinal OCT scans into **CNV, DME, DRUSEN, NORMAL** using PyTorch.
The project includes a NumPy neural-network warm-up, a custom CNN, four transfer-learning models
(ResNet18, VGG16, MobileNetV2, DenseNet121), Grad-CAM explainability and a Streamlit web app.

> **Disclaimer:** This is an educational prototype and **not** a clinical diagnostic tool.

## Files

| File | Purpose |
|------|---------|
| `OCT_Retinal_Disease_Detection.ipynb` | Full workflow: data loading, preprocessing, training, evaluation, Grad-CAM, saving the model |
| `streamlit_app.py` | Web app: upload an OCT image, see prediction, confidence and Grad-CAM |
| `requirements.txt` | Packages needed to run the Streamlit app |
| `best_model.pt` | Created by the notebook (not included in this repository) |

## Dataset

[`zacharielegault/Kermany2017-OCT`](https://huggingface.co/datasets/zacharielegault/Kermany2017-OCT) on Hugging Face
(based on Kermany et al., Cell 2018).

* `train` split: 108,309 images, `test` split: 1,000 images
* Columns: `image` (grayscale OCT scan), `label` (CNV, DME, DRUSEN, NORMAL)
* The dataset is downloaded automatically by the notebook. Do not upload it to GitHub.

## Step 1 - Train the model (notebook)

**Recommended: Google Colab with a GPU** (Runtime -> Change runtime type -> T4 GPU).

1. Upload `OCT_Retinal_Disease_Detection.ipynb` to Colab and run all cells.
2. The last cell saves `best_model.pt` and downloads it.

**VS Code / local computer:**
```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn datasets pillow ipykernel
```
Open the notebook, select the `venv` kernel and run all cells. Without a GPU, reduce
`MAX_TRAIN_PER_CLASS` and `MODELS_TO_TRAIN` in the notebook settings.

## Step 2 - Run the Streamlit app

Put `streamlit_app.py`, `requirements.txt` and `best_model.pt` in the **same folder**, then:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Upload a JPG/PNG OCT image. Test images can be saved from the notebook's test split.

## What the notebook does

1. Loads the dataset and explores class balance, sample images and image sizes
2. Preprocessing: resize 224x224, RGB conversion, augmentation, normalization, DataLoaders
3. NumPy neural network (forward propagation, loss, backpropagation, ReLU/Sigmoid/Tanh)
4. Custom CNN built from scratch
5. Transfer learning: ResNet18, VGG16, MobileNetV2, DenseNet121 (feature extraction, then fine-tuning)
6. Regularization: Dropout, L2 weight decay, BatchNorm, early stopping, hyper-parameter experiment
7. Evaluation on the unseen test split: accuracy, precision, recall, F1, confusion matrix, speed
8. Grad-CAM visualizations
9. Saves the best model (chosen by validation accuracy)

## Notes and limitations

* Trains on a balanced subset by default (1,000 images per class); set `MAX_TRAIN_PER_CLASS = None` for all data.
* The dataset has no patient IDs, so train/validation may contain images of the same patient.
  The official test split is the trustworthy final check.
* Not clinically validated. Performance on scans from other devices or hospitals is unknown.

## Git quick start

```bash
git init
git add OCT_Retinal_Disease_Detection.ipynb streamlit_app.py requirements.txt README.md
git commit -m "Add OCT classification notebook and Streamlit app"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```
Do not commit `best_model.pt` if it is large (use GitHub Releases or Google Drive), and never commit credentials.
