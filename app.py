import os
import json
from pathlib import Path

import gdown
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =========================
# PAGE
# =========================
st.set_page_config(page_title="Fruit Classifier", layout="wide")
st.title("🍎 Fruit Classifier")
st.write("Загрузи изображение, выбери одну или несколько моделей и получи предсказание.")

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(".")
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ВСТАВЬ СВОЙ ID ПАПКИ GOOGLE DRIVE
# У тебя он такой:
GDRIVE_FOLDER_ID = "1yRwfftCC_9Gzuba1n42_cxpP4jrDYuPR"

CLASSES_FILE = MODELS_DIR / "classes.txt"
MODEL_COMPARISON_FILE = MODELS_DIR / "model_comparison.json"

SUPPORTED_MODELS = {
    "alexnet": MODELS_DIR / "alexnet_best_model.pth",
    "resnet50": MODELS_DIR / "resnet50_best_model.pth",
    "vgg16": MODELS_DIR / "vgg16_best_model.pth",
    "googlenet": MODELS_DIR / "googlenet_best_model.pth",
    "efficientnet": MODELS_DIR / "efficientnet_best_model.pth",
}

# =========================
# DOWNLOAD FILES
# =========================
def required_files_present() -> bool:
    needed = [
        CLASSES_FILE,
        MODEL_COMPARISON_FILE,
    ]
    return all(p.exists() for p in needed)

def download_drive_folder_once():
    if required_files_present():
        return

    with st.spinner("Скачиваю файлы из Google Drive... Это может занять время."):
        gdown.download_folder(
            id=GDRIVE_FOLDER_ID,
            output=str(MODELS_DIR),
            quiet=False,
            use_cookies=False,
            remaining_ok=True
        )

download_drive_folder_once()

# =========================
# HELPERS
# =========================
@st.cache_data
def load_classes(classes_file: Path):
    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt not found: {classes_file}")

    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    if not classes:
        raise ValueError("classes.txt is empty")

    return classes


@st.cache_data
def load_model_scores(scores_file: Path):
    if not scores_file.exists():
        return {}

    with open(scores_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_weight(model_name: str, model_scores: dict) -> float:
    if model_name in model_scores:
        info = model_scores[model_name]
        if isinstance(info, dict):
            for key in ["best_accuracy", "accuracy", "val_accuracy", "final_accuracy"]:
                if key in info:
                    try:
                        return float(info[key])
                    except Exception:
                        pass
        elif isinstance(info, (int, float)):
            return float(info)
    return 1.0


def create_model(model_name: str, num_classes: int):
    if model_name == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_name == "googlenet":
        model = models.googlenet(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


@st.cache_resource
def load_model(model_name: str, model_path_str: str, num_classes: int):
    model = create_model(model_name, num_classes)
    state_dict = torch.load(model_path_str, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_single_model(model, image: Image.Image, classes, top_k=5):
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    k = min(top_k, len(classes))
    top_probs, top_indices = torch.topk(probs, k=k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class": classes[idx.item()],
            "confidence": float(prob.item() * 100)
        })

    return probs.cpu(), results


def ensemble_predict(selected_model_names, loaded_models, image, classes, model_scores, top_k=5):
    all_probs = []
    weights = []

    for model_name in selected_model_names:
        probs, _ = predict_single_model(
            model=loaded_models[model_name],
            image=image,
            classes=classes,
            top_k=top_k
        )
        all_probs.append(probs)
        weights.append(get_model_weight(model_name, model_scores))

    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1.0] * len(weights)
        total_weight = sum(weights)

    weighted_probs = sum(p * w for p, w in zip(all_probs, weights)) / total_weight

    k = min(top_k, len(classes))
    top_probs, top_indices = torch.topk(weighted_probs, k=k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class": classes[idx.item()],
            "confidence": float(prob.item() * 100)
        })

    return results


# =========================
# LOAD METADATA
# =========================
try:
    classes = load_classes(CLASSES_FILE)
    num_classes = len(classes)
except Exception as e:
    st.error(f"Ошибка загрузки classes.txt: {e}")
    st.stop()

model_scores = load_model_scores(MODEL_COMPARISON_FILE)

available_model_names = [
    model_name for model_name, model_path in SUPPORTED_MODELS.items()
    if model_path.exists()
]

if not available_model_names:
    st.error("Не найдено ни одной модели .pth в папке models.")
    st.write(f"Проверялась папка: {MODELS_DIR}")
    st.stop()

# =========================
# LOAD MODELS
# =========================
loaded_models = {}
failed_models = {}

with st.spinner("Загружаю модели..."):
    for model_name in available_model_names:
        try:
            loaded_models[model_name] = load_model(
                model_name=model_name,
                model_path_str=str(SUPPORTED_MODELS[model_name]),
                num_classes=num_classes
            )
        except Exception as e:
            failed_models[model_name] = str(e)

if not loaded_models:
    st.error("Ни одна модель не загрузилась.")
    st.json(failed_models)
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙ Настройки")

prediction_mode = st.sidebar.radio(
    "Режим предсказания",
    ["Одна модель", "Несколько моделей (ensemble)"]
)

loaded_model_names = list(loaded_models.keys())

auto_top3 = st.sidebar.checkbox("Автоматически выбрать top-3 лучшие модели", value=False)
top_k = st.sidebar.slider("Количество top predictions", 1, 10, 5)

if auto_top3:
    sorted_models = sorted(
        loaded_model_names,
        key=lambda x: get_model_weight(x, model_scores),
        reverse=True
    )
    selected_models = sorted_models[:min(3, len(sorted_models))]
    prediction_mode = "Несколько моделей (ensemble)"
    st.sidebar.write("Выбраны модели:")
    for m in selected_models:
        st.sidebar.write(f"- {m}")
else:
    if prediction_mode == "Одна модель":
        selected_model = st.sidebar.selectbox("Выбери модель", loaded_model_names)
        selected_models = [selected_model]
    else:
        default_models = loaded_model_names[:min(3, len(loaded_model_names))]
        selected_models = st.sidebar.multiselect(
            "Выбери модели для ensemble",
            loaded_model_names,
            default=default_models
        )

st.sidebar.markdown("### Найденные модели")
for model_name in loaded_model_names:
    weight = get_model_weight(model_name, model_scores)
    st.sidebar.write(f"- {model_name}: weight={weight:.2f}")

if failed_models:
    st.sidebar.markdown("### Не загрузились")
    for model_name in failed_models:
        st.sidebar.write(f"- {model_name}")

# =========================
# MAIN
# =========================
uploaded_file = st.file_uploader(
    "Загрузи изображение",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Загруженное изображение", use_container_width=True)

    with col2:
        if st.button("🔍 Predict", use_container_width=True):
            try:
                if prediction_mode == "Несколько моделей (ensemble)" and len(selected_models) == 0:
                    st.warning("Выбери хотя бы одну модель.")
                else:
                    if prediction_mode == "Одна модель":
                        _, results = predict_single_model(
                            model=loaded_models[selected_models[0]],
                            image=image,
                            classes=classes,
                            top_k=top_k
                        )

                        st.success(f"Использована модель: {selected_models[0]}")
                        st.subheader("Результаты")
                        for i, item in enumerate(results, start=1):
                            st.write(f"{i}. **{item['class']}** — {item['confidence']:.2f}%")

                    else:
                        results = ensemble_predict(
                            selected_model_names=selected_models,
                            loaded_models=loaded_models,
                            image=image,
                            classes=classes,
                            model_scores=model_scores,
                            top_k=top_k
                        )

                        st.success(f"Ensemble из моделей: {', '.join(selected_models)}")
                        st.subheader("Итоговые результаты")
                        for i, item in enumerate(results, start=1):
                            st.write(f"{i}. **{item['class']}** — {item['confidence']:.2f}%")

                        st.markdown("---")
                        st.subheader("Результаты каждой модели отдельно")
                        for model_name in selected_models:
                            _, model_results = predict_single_model(
                                model=loaded_models[model_name],
                                image=image,
                                classes=classes,
                                top_k=min(3, top_k)
                            )
                            st.write(f"**{model_name}**")
                            for item in model_results:
                                st.write(f"- {item['class']} — {item['confidence']:.2f}%")

            except Exception as e:
                st.error(f"Ошибка во время предсказания: {e}")

# =========================
# INFO
# =========================
st.markdown("---")
st.write(f"Папка моделей: `{MODELS_DIR}`")
st.write(f"Используемое устройство: `{DEVICE}`")