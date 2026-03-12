import os
import json
import io
import base64
import time
from datetime import datetime
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
st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="wide")

st.markdown("""
<style>

@import url("https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap");

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
    background: #f5f7f6;
    color: #1c2b1f;
}

/* sidebar */
[data-testid="stSidebar"]{
    background: linear-gradient(160deg,#0a1a0f,#142014);
    border-right:1px solid #2a4a2f;
}

/* upload box */
div[data-testid="stFileUploadDropzone"]{
    background:#ffffff;
    border:2px dashed #9fbfa8;
}

/* result card */
.result-card{
    background:#ffffff;
    border-radius:14px;
    padding:25px;
    border:1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(".")
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

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

EMOJIS = {
    "apple": "🍎", "banana": "🍌", "broccoli": "🥦", "carrot": "🥕", "cucumber": "🥒",
    "date_plum": "🫐", "garlic": "🧄", "grape": "🍇", "kiwi": "🥝", "lemon": "🍋",
    "mandarin": "🍊", "onion": "🧅", "orange": "🍊", "pear": "🍐", "persimmon": "🍅",
    "pineapple": "🍍", "pitahaya": "🐉", "pomegranate": "🍎", "potato": "🥔",
    "quince": "🍐", "strawberry": "🍓"
}

if "history" not in st.session_state:
    st.session_state.history = []

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
with st.sidebar:
    st.markdown("### 🌿 Fruit Classifier")
    st.markdown("---")

    prediction_mode = st.radio(
        "Режим предсказания",
        ["Одна модель", "Несколько моделей (ensemble)"]
    )

    loaded_model_names = list(loaded_models.keys())
    auto_top3 = st.checkbox("Автоматически выбрать top-3 лучшие модели", value=False)
    top_k = st.slider("Количество top predictions", 1, 10, 5)

    if auto_top3:
        sorted_models = sorted(
            loaded_model_names,
            key=lambda x: get_model_weight(x, model_scores),
            reverse=True
        )
        selected_models = sorted_models[:min(3, len(sorted_models))]
        prediction_mode = "Несколько моделей (ensemble)"
        st.markdown("**Выбраны модели:**")
        for m in selected_models:
            st.markdown(f"- {m}")
    else:
        if prediction_mode == "Одна модель":
            selected_model = st.selectbox("Выбери модель", loaded_model_names)
            selected_models = [selected_model]
        else:
            default_models = loaded_model_names[:min(3, len(loaded_model_names))]
            selected_models = st.multiselect(
                "Выбери модели для ensemble",
                loaded_model_names,
                default=default_models
            )

    st.markdown("---")
    st.markdown("**📋 Доступные классы:**")
    for c in classes:
        st.markdown(f"{EMOJIS.get(c, '🌿')} {c.replace('_', ' ').title()}")

    st.markdown("---")
    st.markdown("**⚖️ Веса моделей:**")
    for model_name in loaded_model_names:
        weight = get_model_weight(model_name, model_scores)
        st.markdown(f"- {model_name}: `{weight:.2f}`")

    if failed_models:
        st.markdown("---")
        st.markdown("**Не загрузились:**")
        for model_name in failed_models:
            st.markdown(f"- {model_name}")

    if st.button("🗑️ Очистить историю", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# =========================
# HEADER
# =========================
st.markdown('<p style="font-family:Playfair Display,serif;font-size:2.9rem;font-weight:700;background:linear-gradient(135deg,#7dde7d,#ffd580);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.2rem">Fruit Classifier 🍃</p>', unsafe_allow_html=True)
st.markdown("Загрузи фото фрукта или овоща — получи предсказание модели или ensemble.")

m1, m2, m3 = st.columns(3)
m1.metric("Активных моделей", len(loaded_models))
m2.metric("Режим", "Ensemble" if prediction_mode == "Несколько моделей (ensemble)" else "Single")
m3.metric("Устройство", str(DEVICE).upper())

# =========================
# MAIN
# =========================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Загрузи изображение")
    uploaded_file = st.file_uploader(
        "Загрузи изображение",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженное изображение", use_container_width=True)
        predict_btn = st.button("🔍 Predict", use_container_width=True)
    else:
        st.markdown('<div style="text-align:center;padding:3rem;color:#4a7a4a;border:2px dashed #2a4a2f;border-radius:12px;background:#132217"><div style="font-size:3rem">🍎</div><div>Загрузи фото для анализа</div></div>', unsafe_allow_html=True)
        image = None
        predict_btn = False

with col2:
    st.markdown("### 🧠 Результат")

    if image is not None and predict_btn:
        try:
            if prediction_mode == "Несколько моделей (ensemble)" and len(selected_models) == 0:
                st.warning("Выбери хотя бы одну модель.")
            else:
                started = time.time()

                if prediction_mode == "Одна модель":
                    _, results = predict_single_model(
                        model=loaded_models[selected_models[0]],
                        image=image,
                        classes=classes,
                        top_k=top_k
                    )
                    status_text = f"Использована модель: {selected_models[0]}"
                else:
                    results = ensemble_predict(
                        selected_model_names=selected_models,
                        loaded_models=loaded_models,
                        image=image,
                        classes=classes,
                        model_scores=model_scores,
                        top_k=top_k
                    )
                    status_text = f"Ensemble из моделей: {', '.join(selected_models)}"

                elapsed = time.time() - started
                top = results[0]
                top_class = top["class"]
                top_emoji = EMOJIS.get(top_class, "🌿")

                st.success(status_text)
                st.markdown(
                    f'''<div style="background:linear-gradient(135deg,#162d1a,#1e3a22);border:1px solid #3a6040;border-radius:16px;padding:1.8rem;margin-top:1rem">
                    <div style="font-size:3rem">{top_emoji}</div>
                    <div style="font-family:Playfair Display,serif;font-size:2rem;color:#a8f0a8;font-weight:700">{top_class.replace('_',' ').title()}</div>
                    <div style="color:#8aaa8a;font-size:0.8rem;margin-top:1rem;text-transform:uppercase">Уверенность</div>
                    <div style="font-size:1.8rem;color:#7dde7d;font-weight:700">{top['confidence']:.2f}%</div>
                    <div style="font-size:0.75rem;color:#4a7a4a;margin-top:0.5rem">⚡ {elapsed*1000:.0f} ms</div>
                    </div>''',
                    unsafe_allow_html=True
                )

                st.markdown("**Top predictions:**")
                for item in results:
                    pct = max(0.0, min(item["confidence"] / 100.0, 1.0))
                    st.markdown(f"{EMOJIS.get(item['class'], '🌿')} **{item['class'].replace('_',' ').title()}** — {item['confidence']:.2f}%")
                    st.progress(pct)

                if prediction_mode == "Несколько моделей (ensemble)":
                    st.markdown("---")
                    st.markdown("### Результаты каждой модели отдельно")
                    for model_name in selected_models:
                        _, model_results = predict_single_model(
                            model=loaded_models[model_name],
                            image=image,
                            classes=classes,
                            top_k=min(3, top_k)
                        )
                        st.markdown(f"**{model_name}**")
                        for item in model_results:
                            st.markdown(f"- {item['class']} — {item['confidence']:.2f}%")

                thumb = image.copy()
                thumb.thumbnail((120, 120))
                buf = io.BytesIO()
                thumb.save(buf, format="JPEG")
                st.session_state.history.insert(0, {
                    "fruit": top_class,
                    "mode": prediction_mode,
                    "model": ", ".join(selected_models),
                    "conf": top["confidence"],
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "thumb": base64.b64encode(buf.getvalue()).decode()
                })

        except Exception as e:
            st.error(f"Ошибка во время предсказания: {e}")
    else:
        st.markdown('<div style="text-align:center;padding:4rem;color:#4a7a4a"><div style="font-size:2.5rem">🔬</div><div>Результат появится здесь</div></div>', unsafe_allow_html=True)

# =========================
# FOOTER INFO
# =========================
st.markdown("---")
st.markdown("### 🧩 Информация")
i1, i2 = st.columns(2)
with i1:
    st.markdown(f'<div class="small-card"><div style="color:#9bc79d">Папка моделей</div><div style="font-weight:600">{MODELS_DIR}</div></div>', unsafe_allow_html=True)
with i2:
    st.markdown(f'<div class="small-card"><div style="color:#9bc79d">Устройство</div><div style="font-weight:600">{DEVICE}</div></div>', unsafe_allow_html=True)

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕑 История определений")
    hcols = st.columns(4)
    for idx, entry in enumerate(st.session_state.history[:8]):
        hcols[idx % 4].markdown(
            f'''<div style="background:#162d1a;border:1px solid #2a4a2f;border-radius:12px;overflow:hidden;margin-bottom:0.5rem">
            <img src="data:image/jpeg;base64,{entry['thumb']}" style="width:100%;height:90px;object-fit:cover">
            <div style="padding:0.6rem">
            <div style="color:#a8f0a8;font-weight:500">{EMOJIS.get(entry['fruit'],'🌿')} {entry['fruit'].replace('_',' ').title()}</div>
            <div style="font-size:0.72rem;color:#6a8a6a">{entry['conf']:.2f}%</div>
            <div style="font-size:0.68rem;color:#4a6a4a">{entry['model']} · {entry['time']}</div>
            </div></div>''',
            unsafe_allow_html=True
        )
