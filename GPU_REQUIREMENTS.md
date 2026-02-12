# GPU_REQUIREMENTS

Bu proje GPU'yu Python tarafinda **yalnizca embedding** (SentenceTransformers / PyTorch) icin kullanir.
Gemini LLM/VLM API cagirilari uzak servis oldugu icin GPU ile hizlanmaz.

> Not (Local mod): `LLM_PROVIDER=local` / `VLM_PROVIDER=local` iken Ollama kendi surecinde GPU kullanir.
> Bu durumda VRAM kapasitesi local LLM/VLM performansi icin kritik hale gelir.

## Ne kazandirir?

- PDF'ler ilk kez indexlenirken (chunk embedding) ve retrieval sirasinda query embedding uretilirken hizlanma.

## Local LLM/VLM (Ollama) icin VRAM notu

- VRAM, local modelin GPU'da calisip calisamayacagini belirler.
- Daha buyuk model (veya daha uzun context) daha fazla VRAM ister.
- VRAM yetersizse CPU offload olabilir; bu da token hizini ve cevap gecikmesini belirgin etkiler.
- Pratikte 6 GB VRAM sinifinda 7B quantize modeller daha guvenli bir denge verir.

## Model boyutu, quantization ve OOM (Local LLM/VLM)

- Bu repo'nun referans GPU'su (GTX 1660 SUPER, 6 GB VRAM) dusuk/orta seviye oldugu icin, local modda amac "en buyuk modeli kosmak" degil; **tam offline calisma** yetenegini gostermektir.
- Buyuk acik modellerde (ornegin 70B/120B sinifi; `gpt-oss-120b` gibi) quantization genellikle zorunlu hale gelir. Aksi halde model agirliklari VRAM/RAM'e sigmayabilir.
- OOM (out-of-memory) sadece model agirliklarindan kaynaklanmaz:
  - Context uzadiginda **KV cache** bellek kullanimi artar.
  - Eszamanli istek/batch arttikca bellek baskisi artar.
- OOM durumunda tipik sonuc:
  - Model yuklenemez veya inference sirasinda hata verir
  - veya CPU offload ile calisir (calisabilir ama hiz/latency ciddi etkilenir)

## Kurulum (Windows, onerilen yaklasim)

GPU kurulumunu CPU ortamindan ayri tutmak (ayri venv) en az sorunlu yaklasimdir.

### 1) GPU venv olustur

```bash
cd <repo_root>
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
python -c "from src.config import load_settings; from src.core.embedding import Embedder; s=load_settings(); print('selected', s.embedding_model, s.embedding_device); e=Embedder(s.embedding_model, device=s.embedding_device); e.embed_query('test'); print('device:', e._model.device)"
```

> Not: Yanlis venv ile calistirmamak icin app'i su sekilde baslatmak en garantisi:
>
> `.\.venv-gpu\Scripts\python.exe -m chainlit run app.py -w`

## Opsiyonel: Embedding device secimi

Varsayilan: `EMBEDDING_DEVICE=auto` (CUDA varsa GPU, yoksa CPU).

`.env` icinde override edebilirsiniz:

```ini
EMBEDDING_DEVICE=auto
# EMBEDDING_DEVICE=cpu
# EMBEDDING_DEVICE=cuda
```

## Troubleshooting

- **CUDA True ama hizlanma yok**: embedding device kontrol komutuyla `device: cuda` gorundugunu teyit edin.
- **Port cakismasi (8000)**: `README.md` → Troubleshooting bolumune bakin (`--port 8001` veya `AUTO_EXIT_ON_NO_CLIENTS=1`).

