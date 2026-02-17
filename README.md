# Product Color Extractor

Backend service for object-aware dominant color extraction using
depth, saliency.

## Setup

```bash
git clone <repo-url>
cd product-color-extractor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

Download U²Net Lite weights:

```bash
pip install gdown
gdown "https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy" -O weights/u2netp.pth
```

Run server:

```bash
uvicorn backend.app:app --reload --reload-dir backend
```

Run UI:

```bash
streamlit run ui/app.py
```

Open:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
