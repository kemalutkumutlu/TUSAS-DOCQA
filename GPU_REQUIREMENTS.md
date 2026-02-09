# GPU_REQUIREMENTS

Bu proje GPU'yu **yalnizca embedding** (SentenceTransformers / PyTorch) tarafinda kullanir.
Gemini LLM/VLM API cagirilari uzak servis oldugu icin GPU ile hizlanmaz.

## Ne kazandirir?

- PDF'ler ilk kez indexlenirken (chunk embedding) ve retrieval sirasinda query embedding uretilirken hizlanma.

## Kurulum (Windows, onerilen yaklasim)

GPU kurulumunu CPU ortamindan ayri tutmak (ayri venv) en az sorunlu yaklasimdir.

### 1) GPU venv olustur

```bash
cd C:\Users\Kemal\Desktop\TUSAS-DOCQA-YENI
python -m venv .venv-gpu
.\.venv-gpu\Scripts\activate
python -m pip install -U pip
```

### 2) CUDA uyumlu PyTorch kur

Her kullanicinin CUDA/driver uyumu farkli olabildigi icin PyTorch paketini uygun CUDA index-url ile kurun.
Ornek (CUDA 12.1):

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3) Proje bagimliliklarini kur

```bash
python -m pip install -r requirements.txt
```

## Dogrulama

### Torch + GPU goruyor mu?

```bash
python -c "import torch; print('torch',torch.__version__); print('torch_cuda',torch.version.cuda); print('cuda_available',torch.cuda.is_available()); print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### Embedding gercekten GPU'da mi?

```bash
python -c "from src.core.embedding import Embedder; e=Embedder('intfloat/multilingual-e5-small'); e.embed_query('test'); print('device:', e._model.device)"
```

## Troubleshooting

- **CUDA True ama hizlanma yok**: embedding device kontrol komutuyla `device: cuda` gorundugunu teyit edin.
- **Port cakismasi (8000)**: `README.md` → Troubleshooting bolumune bakin (`--port 8001` veya `AUTO_EXIT_ON_NO_CLIENTS=1`).

