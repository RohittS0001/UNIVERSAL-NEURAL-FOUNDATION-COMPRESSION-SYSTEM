import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import os
import struct
import hashlib
import time
import threading
import warnings
import json
import sys
import cv2
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False
try:
    from moviepy.editor import ImageSequenceClip
    MOVIEPY_AVAILABLE = True
except Exception:
    try:
        from moviepy import ImageSequenceClip
        MOVIEPY_AVAILABLE = True
    except Exception:
        MOVIEPY_AVAILABLE = False
import PyPDF2
import docx as docx_lib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

warnings.filterwarnings('ignore')

# ============================================
# UNIVERSAL NEURAL FOUNDATION v5.3
# ALL 13 CLAIMS + ALL FILE TYPES
# Images, Video, Audio, Documents, Any file
# Synced with train_foundation.py v5.6
# Inventor: Rohit Kalu Sasane, Pune India 2026
# ============================================

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS      = "foundation_v4_weights.pth"
MAGIC        = b'UNFC'
FRAME_SAMPLE = 3    # compress every Nth frame
MAX_PARENT_SCAN = 50  # max DNA files to scan for inheritance

IMAGE_EXT    = {'jpg','jpeg','png','bmp','webp','gif','tiff'}
VIDEO_EXT    = {'mp4','avi','mov','mkv','wmv','flv'}
AUDIO_EXT    = {'mp3','wav','flac','aac','ogg','m4a'}
DOCUMENT_EXT = {'pdf','docx','txt','csv','doc'}


# ── ARCHITECTURE — must match train_foundation.py ──

class GlobalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(128*4*4, 32)
        )
    def forward(self, x): return self.net(x)


class RegionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(256*4*4, 256)
        )
    def forward(self, x): return self.net(x)


class DetailEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(8), nn.Flatten(),
            nn.Linear(512*8*8, 512)
        )
    def forward(self, x): return self.net(x)


class MorphogeneticDecoder(nn.Module):
    """Claim 12"""
    def __init__(self):
        super().__init__()
        self.combine = nn.Sequential(
            nn.Linear(800, 512*8*8), nn.ReLU()
        )
        self.field_generator = nn.Linear(512*8*8, 64)
        self.field_receiver  = nn.Linear(64, 512*8*8)
        self.decode = nn.Sequential(
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,  3,   4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, g, r, d):
        x = self.combine(torch.cat([g, r, d], dim=1))
        x = x + 0.1 * self.field_receiver(self.field_generator(x))
        return self.decode(x)


class Foundation(nn.Module):
    """Claims 8 9 12"""
    PATHWAYS = ['face','landscape','illustration','document','abstract']

    def __init__(self):
        super().__init__()
        self.global_encoder    = GlobalEncoder()
        self.regional_encoder  = RegionalEncoder()
        self.detail_encoder    = DetailEncoder()
        self.decoder           = MorphogeneticDecoder()
        self.pathway_classifier = nn.Sequential(
            nn.Linear(800, 256), nn.ReLU(), nn.Linear(256, 5)
        )

    def forward(self, x):
        g = self.global_encoder(x)
        r = self.regional_encoder(x)
        d = self.detail_encoder(x)
        return self.decoder(g, r, d), g, r, d

    def encode(self, x):
        with torch.no_grad():
            return (self.global_encoder(x),
                    self.regional_encoder(x),
                    self.detail_encoder(x))

    def decode(self, g, r, d):
        with torch.no_grad():
            return self.decoder(g, r, d)

    def get_pathway(self, g, r, d):
        with torch.no_grad():
            pid = torch.argmax(
                self.pathway_classifier(
                    torch.cat([g, r, d], dim=1)
                ), dim=1
            ).item()
        return pid, self.PATHWAYS[pid]


# ── ENCRYPTION — Claim 6 ─────────────────────

class DNA_Enc:
    def __init__(self):
        import base64 as b64
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(), length=32,
            salt=b'rohit_sasane_pune_india_2026',
            iterations=100000
        )
        self.c = Fernet(b64.urlsafe_b64encode(
            kdf.derive(b'unfcs-rohit-2026')
        ))
    def enc(self, d): return self.c.encrypt(d)
    def dec(self, d): return self.c.decrypt(d)


# ── DNA FORMAT — Claims 6+8 ──────────────────

def save_dna(path, g, r, d, checksum,
             pid=0, parent=None, encrypt=True,
             file_type='image', meta=None):
    raw = struct.pack('HHH', 32, 256, 512)
    for arr in [g, r, d]:
        raw += arr.numpy().flatten().astype(np.float32).tobytes()
    if encrypt:
        raw = DNA_Enc().enc(raw)
    meta_bytes = json.dumps(meta or {}).encode()
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('BBB', 3, int(encrypt), pid))
        f.write(checksum.encode()[:32].ljust(32, b'\x00'))
        f.write(file_type.encode()[:16].ljust(16, b'\x00'))
        f.write(struct.pack('I', len(raw)))
        f.write(raw)
        f.write(struct.pack('I', len(meta_bytes)))
        f.write(meta_bytes)
        if parent:
            pb = parent.encode()
            f.write(struct.pack('BH', 1, len(pb)))
            f.write(pb)
        else:
            f.write(struct.pack('B', 0))


def load_dna(path):
    with open(path, 'rb') as f:
        assert f.read(4) == MAGIC, "Invalid DNA file"
        _, enc_flag, pid = struct.unpack('BBB', f.read(3))
        checksum = f.read(32).rstrip(b'\x00').decode()
        file_type = f.read(16).rstrip(b'\x00').decode()
        raw = f.read(struct.unpack('I', f.read(4))[0])
        if enc_flag:
            raw = DNA_Enc().dec(raw)
        dims = struct.unpack('HHH', raw[:6])
        off  = 6
        vecs = []
        for dim in dims:
            vecs.append(
                np.frombuffer(raw[off:off+dim*4],
                              dtype=np.float32).copy()
            )
            off += dim*4
        meta_len = struct.unpack('I', f.read(4))[0]
        meta     = json.loads(f.read(meta_len).decode())
        has_p    = struct.unpack('B', f.read(1))[0]
        parent   = None
        if has_p:
            parent = f.read(
                struct.unpack('H', f.read(2))[0]
            ).decode()
    return (
        torch.FloatTensor(vecs[0]).unsqueeze(0),
        torch.FloatTensor(vecs[1]).unsqueeze(0),
        torch.FloatTensor(vecs[2]).unsqueeze(0),
        checksum, pid, parent, file_type, meta
    )


# ── CLAIM 7 — INHERITANCE ────────────────────

def find_parent(g, r, d, folder=".", threshold=0.85):
    """
    Find similar existing DNA file.
    Limited to MAX_PARENT_SCAN files for speed.
    """
    best_sim, best = 0, None
    v1 = torch.cat([g.flatten(), r.flatten(), d.flatten()])

    # Only scan .dna files — not .vdna or .adna
    dna_files = [
        f for f in os.listdir(folder)
        if f.endswith('.dna') and not f.endswith('.dna.txt')
    ]
    # Limit scan for performance
    if len(dna_files) > MAX_PARENT_SCAN:
        dna_files = dna_files[:MAX_PARENT_SCAN]

    for fname in dna_files:
        try:
            result = load_dna(os.path.join(folder, fname))
            pg, pr, pd = result[0], result[1], result[2]
            v2 = torch.cat([
                pg.flatten(), pr.flatten(), pd.flatten()
            ])
            s = nn.functional.cosine_similarity(
                v1.unsqueeze(0), v2.unsqueeze(0)
            ).item()
            if s > best_sim:
                best_sim, best = s, fname
        except:
            continue
    return (best, best_sim) if best_sim > threshold else (None, 0)


# ── EPIGENETIC MODES — Claim 10 ──────────────

MODES = {
    "thumbnail": {"size": 64,  "chains": "g",
                  "desc": "32 numbers — instant thumbnail"},
    "mobile":    {"size": 128, "chains": "gr",
                  "desc": "288 numbers — mobile quality"},
    "full":      {"size": 256, "chains": "grd",
                  "desc": "800 numbers — full quality"},
    "print":     {"size": 512, "chains": "grd",
                  "desc": "800 numbers — print quality"},
}


# ── CLAIM 4 — FEDERATED ──────────────────────

class Federated:
    def __init__(self, f):
        self.f = f
        self.grads = []

    def collect(self, t):
        self.f.train()
        out, *_ = self.f(t)
        loss = nn.MSELoss()(out, t)
        loss.backward()
        g = {
            n: (p.grad + torch.normal(0,.001,p.grad.shape)).detach()
            for n, p in self.f.named_parameters()
            if p.grad is not None
        }
        self.f.zero_grad()
        self.f.eval()
        self.grads.append(g)

    def aggregate(self):
        if not self.grads:
            return
        n = len(self.grads)
        with torch.no_grad():
            for name, p in self.f.named_parameters():
                if name in self.grads[0]:
                    p -= 0.0001 * sum(
                        g[name] for g in self.grads
                    ) / n
        self.grads = []
        print(f"  Federated: {n} gradients aggregated")


# ── CLAIM 11 — MICRO-LEARNING ────────────────

class MicroLearner:
    def __init__(self, f):
        self.f   = f
        self.opt = torch.optim.Adam(f.parameters(), lr=1e-5)
        self.n   = 0

    def update(self, t):
        self.f.train()
        out, *_ = self.f(t)
        loss = nn.MSELoss()(out, t)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.f.parameters(), 0.01)
        self.opt.step()
        self.f.eval()
        self.n += 1


# ── CLAIM 13 — THREE TIER ────────────────────

def deploy_tiers(weights):
    f = Foundation()
    f.load_state_dict(torch.load(
        weights, weights_only=True, map_location='cpu'
    ))
    q = torch.quantization.quantize_dynamic(
        f, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    torch.save(q.state_dict(), "tier1_device.pth")
    torch.save(f.state_dict(), "tier2_edge.pth")
    t1 = os.path.getsize("tier1_device.pth")
    t2 = os.path.getsize("tier2_edge.pth")
    t3 = os.path.getsize(weights)
    print(f"  Tier 1 Device: {t1/1024/1024:.1f}MB — phone")
    print(f"  Tier 2 Edge:   {t2/1024/1024:.1f}MB — server")
    print(f"  Tier 3 Core:   {t3/1024/1024:.1f}MB — master")
    for tf in ["tier1_device.pth", "tier2_edge.pth"]:
        if os.path.exists(tf):
            os.remove(tf)


# ── STREAMING PIPELINE ───────────────────────

class StreamPipeline:
    def __init__(self, f, workers=4):
        self.f = f
        self.w = workers

    def run(self, paths):
        results = []
        lock    = threading.Lock()

        def worker(p):
            try:
                img = Image.open(p).convert('RGB').resize((256,256))
                t   = torch.FloatTensor(
                    np.array(img)/255.0
                ).permute(2,0,1).unsqueeze(0).to(device)
                with torch.no_grad():
                    g = self.f.global_encoder(t)
                    r = self.f.regional_encoder(t)
                    d = self.f.detail_encoder(t)
                with lock:
                    results.append((p,g.cpu(),r.cpu(),d.cpu()))
            except:
                pass

        threads = []
        for p in paths:
            th = threading.Thread(target=worker, args=(p,))
            threads.append(th)
            th.start()
            if len(threads) >= self.w:
                for th in threads: th.join()
                threads = []
        for th in threads: th.join()
        return results


# ── FILE TYPE HELPERS ─────────────────────────

def frame_to_tensor(frame, size=256):
    img = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ).resize((size, size))
    return torch.FloatTensor(
        np.array(img)/255.0
    ).permute(2,0,1).unsqueeze(0).to(device)


def audio_to_chunks(audio_path, chunk_ms=1000):
    if not PYDUB_AVAILABLE:
        return [], 44100, 1
    audio  = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_ms):
        chunks.append(audio[i:i+chunk_ms])
    return chunks, audio.frame_rate, audio.channels


def chunk_to_image(chunk, size=256):
    """
    Convert audio chunk to RGB spectrogram image.
    Uses all 3 channels for richer representation.
    """
    if not PYDUB_AVAILABLE:
        return torch.zeros(1,3,size,size).to(device)
    samples = np.array(
        chunk.get_array_of_samples()
    ).astype(np.float32)
    if len(samples) == 0:
        samples = np.zeros(size*size)
    samples = samples / (np.max(np.abs(samples)) + 1e-8)
    samples = np.resize(samples, size*size)
    data    = samples.reshape(size, size)
    # Use all 3 channels — more information for foundation
    r_ch = ((data + 1) / 2 * 255).astype(np.uint8)
    g_ch = (np.abs(data) * 255).astype(np.uint8)
    b_ch = np.clip(
        (1 - np.abs(data)) * 180 +
        np.abs(np.roll(data, 8, axis=1)) * 75,
        0, 255
    ).astype(np.uint8)
    img = Image.fromarray(
        np.stack([r_ch, g_ch, b_ch], axis=2)
    )
    return torch.FloatTensor(
        np.array(img)/255.0
    ).permute(2,0,1).unsqueeze(0).to(device)


def extract_document_text(path):
    ext  = path.split('.')[-1].lower()
    text = ""
    try:
        if ext == 'pdf':
            reader = PyPDF2.PdfReader(path)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif ext == 'docx':
            doc = docx_lib.Document(path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif ext in ['txt','csv']:
            with open(path, 'r', encoding='utf-8',
                      errors='ignore') as f:
                text = f.read()
    except Exception as e:
        text = f"extraction_failed: {str(e)}"
    return text


def text_to_image(text, size=256):
    chars = np.array(
        [ord(c) for c in text[:size*size]],
        dtype=np.float32
    )
    if len(chars) < size*size:
        chars = np.pad(chars, (0, size*size - len(chars)))
    chars   = chars / 128.0 - 1.0
    img_arr = (
        (chars[:size*size].reshape(size,size) + 1)
        / 2 * 255
    ).astype(np.uint8)
    img = Image.fromarray(img_arr).convert('RGB')
    return torch.FloatTensor(
        np.array(img)/255.0
    ).permute(2,0,1).unsqueeze(0).to(device)


def tensor_to_image(tensor, size=None):
    """Convert tensor to PIL Image"""
    arr = (
        torch.clamp(tensor, 0, 1)
        .squeeze(0).permute(1,2,0)
        .cpu().numpy() * 255
    ).astype(np.uint8)
    img = Image.fromarray(arr)
    if size:
        img = img.resize((size, size))
    return img


def detect_file_type(path):
    """Auto detect file type from extension"""
    ext = path.split('.')[-1].lower()
    if ext in IMAGE_EXT:    return 'image'
    if ext in VIDEO_EXT:    return 'video'
    if ext in AUDIO_EXT:    return 'audio'
    if ext in DOCUMENT_EXT: return 'document'
    return 'binary'


# ── COMPARISON CANVAS ─────────────────────────

def make_comparison(original_path, recon_tensor,
                    stage1_tensor, stage2_tensor,
                    stage3_tensor, out_path):
    """
    Build side by side comparison:
    Original | Global-32 | Regional-288 | Full-800 | Reconstructed
    """
    size = 256
    orig = Image.open(original_path).convert('RGB').resize((size,size))
    s1   = tensor_to_image(stage1_tensor, size)
    s2   = tensor_to_image(stage2_tensor, size)
    s3   = tensor_to_image(stage3_tensor, size)
    rec  = tensor_to_image(recon_tensor,  size)

    cols    = 5
    padding = 10
    label_h = 30
    w       = cols * size + (cols+1) * padding
    h       = size + label_h + padding * 2

    canvas = Image.new('RGB', (w, h), (15, 15, 25))
    draw   = ImageDraw.Draw(canvas)

    items = [
        (orig, "Original",        (255,255,255)),
        (s1,   "Global (32)",     (150,150,255)),
        (s2,   "Regional (288)",  (150,220,255)),
        (s3,   "Full (800)",      (100,255,150)),
        (rec,  "Reconstructed",   (80, 255, 80)),
    ]

    for i, (img, label, color) in enumerate(items):
        x = padding + i * (size + padding)
        canvas.paste(img, (x, label_h + padding))
        # Center label above image
        draw.text(
            (x + size//2, padding//2 + 5),
            label, fill=color, anchor="mt"
        )

    draw.text(
        (w//2, h - padding//2 - 5),
        "Universal Neural Foundation — Rohit Sasane 2026",
        fill=(80,80,80), anchor="mb"
    )
    canvas.save(out_path)
    return out_path


# ── MAIN SYSTEM ──────────────────────────────

class System:
    def __init__(self):
        self.f       = Foundation()
        self.loss_fn = nn.MSELoss()

        if not os.path.exists(WEIGHTS):
            raise FileNotFoundError(
                f"No weights found: {WEIGHTS}\n"
                f"Run: python train_foundation.py images"
            )

        self.f.load_state_dict(torch.load(
            WEIGHTS, weights_only=True, map_location=device
        ))
        self.f = self.f.to(device)
        self.f.eval()
        self.ml = MicroLearner(self.f)

        sz = os.path.getsize(WEIGHTS)
        print(f"Foundation loaded: {sz/1024/1024:.1f}MB  "
              f"device={device}")
        if not PYDUB_AVAILABLE:
            print("Audio: disabled — pip install audioop-lts")
        if not MOVIEPY_AVAILABLE:
            print("Video reconstruction: disabled — "
                  "pip install moviepy==1.0.3")

    def img_tensor(self, path, size=256):
        return torch.FloatTensor(
            np.array(
                Image.open(path).convert('RGB').resize((size,size))
            )/255.0
        ).permute(2,0,1).unsqueeze(0).to(device)

    # ── COMPRESS ANY FILE ─────────────────────

    def compress(self, path, encrypt=True):
        """
        Auto detect file type and compress to DNA.
        Returns DNA file path.
        """
        if not os.path.exists(path):
            print(f"  File not found: {path}")
            return None

        ext      = path.split('.')[-1].lower()
        ftype    = detect_file_type(path)
        orig     = os.path.getsize(path)

        print(f"\n{'='*50}")
        print(f"  File:  {os.path.basename(path)}")
        print(f"  Type:  {ftype.upper()}  ({ext})")
        print(f"  Size:  {orig/1024/1024:.3f} MB")
        print(f"{'='*50}")

        if ftype == 'image':
            return self._compress_image(path, encrypt, orig)
        elif ftype == 'video':
            return self._compress_video(path, encrypt, orig)
        elif ftype == 'audio':
            return self._compress_audio(path, encrypt, orig)
        elif ftype == 'document':
            return self._compress_document(path, encrypt, orig)
        else:
            return self._compress_binary(path, encrypt, orig)

    def _compress_image(self, path, encrypt, orig):
        t           = self.img_tensor(path)
        g, r, d     = self.f.encode(t)
        gc, rc, dc  = g.cpu(), r.cpu(), d.cpu()
        pid, pname  = self.f.get_pathway(gc, rc, dc)
        parent, sim = find_parent(gc, rc, dc)
        self.ml.update(t)
        recon       = self.f.decode(g, r, d)
        loss        = self.loss_fn(recon, t).item()
        checksum    = hashlib.md5(open(path,'rb').read()).hexdigest()
        dna         = path + ".dna"
        save_dna(
            dna, gc.squeeze(0), rc.squeeze(0), dc.squeeze(0),
            checksum, pid, parent, encrypt, 'image'
        )
        dsz = os.path.getsize(dna)
        print(f"  DNA:      {dsz/1024:.1f} KB  "
              f"({(1-dsz/orig)*100:.2f}% smaller)")
        print(f"  Loss:     {loss:.6f}")
        print(f"  Pathway:  {pname}")
        if parent:
            print(f"  Parent:   {parent}  sim={sim:.3f}")
        return dna

    def _compress_video(self, path, encrypt, orig):
        if not cv2:
            print("  cv2 not available — cannot compress video")
            return None
        cap       = cv2.VideoCapture(path)
        fps       = cap.get(cv2.CAP_PROP_FPS)
        total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration  = total / fps if fps > 0 else 0
        frame_dnas = []
        frame_idx  = 0
        compressed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FRAME_SAMPLE == 0:
                t       = frame_to_tensor(frame)
                g, r, d = self.f.encode(t)
                frame_dnas.append({
                    'g':   g.cpu().squeeze(0).numpy().flatten().tolist(),
                    'r':   r.cpu().squeeze(0).numpy().flatten().tolist(),
                    'd':   d.cpu().squeeze(0).numpy().flatten().tolist(),
                    'idx': frame_idx
                })
                compressed += 1
                if compressed % 50 == 0:
                    print(f"  Frames: {compressed} / "
                          f"{total//FRAME_SAMPLE} compressed")
            frame_idx += 1
        cap.release()

        vdna = path + ".vdna"
        meta = {
            'fps':           fps,
            'total_frames':  total,
            'sampled_frames': compressed,
            'sample_rate':   FRAME_SAMPLE,
            'duration':      duration
        }
        raw = json.dumps(
            {'meta': meta, 'frames': frame_dnas}
        ).encode()
        if encrypt:
            raw = DNA_Enc().enc(raw)

        checksum = hashlib.md5(open(path,'rb').read()).hexdigest()
        with open(vdna, 'wb') as f:
            f.write(b'VDNA')
            f.write(struct.pack('B', int(encrypt)))
            f.write(checksum.encode()[:32].ljust(32, b'\x00'))
            f.write(struct.pack('I', len(raw)))
            f.write(raw)

        dsz = os.path.getsize(vdna)
        print(f"  Frames:   {total} total → {compressed} compressed")
        print(f"  Duration: {duration:.1f}s at {fps:.0f}fps")
        print(f"  DNA:      {dsz/1024:.1f} KB  "
              f"({(1-dsz/orig)*100:.2f}% smaller)")
        return vdna

    def _compress_audio(self, path, encrypt, orig):
        if not PYDUB_AVAILABLE:
            print("  Audio skipped — install audioop-lts")
            return None
        chunks, sample_rate, channels = audio_to_chunks(path)
        chunk_dnas = []
        for chunk in chunks:
            t       = chunk_to_image(chunk)
            g, r, d = self.f.encode(t)
            chunk_dnas.append({
                'g':           g.cpu().squeeze(0).numpy().flatten().tolist(),
                'r':           r.cpu().squeeze(0).numpy().flatten().tolist(),
                'd':           d.cpu().squeeze(0).numpy().flatten().tolist(),
                'duration_ms': len(chunk)
            })
        adna = path + ".adna"
        meta = {
            'sample_rate': sample_rate,
            'channels':    channels,
            'chunks':      len(chunk_dnas),
            'format':      path.split('.')[-1]
        }
        raw = json.dumps(
            {'meta': meta, 'chunks': chunk_dnas}
        ).encode()
        if encrypt:
            raw = DNA_Enc().enc(raw)
        checksum = hashlib.md5(open(path,'rb').read()).hexdigest()
        with open(adna, 'wb') as f:
            f.write(b'ADNA')
            f.write(struct.pack('B', int(encrypt)))
            f.write(checksum.encode()[:32].ljust(32, b'\x00'))
            f.write(struct.pack('I', len(raw)))
            f.write(raw)
        dsz = os.path.getsize(adna)
        print(f"  Chunks:   {len(chunk_dnas)} seconds")
        print(f"  DNA:      {dsz/1024:.1f} KB  "
              f"({(1-dsz/orig)*100:.2f}% smaller)")
        return adna

    def _compress_document(self, path, encrypt, orig):
        text        = extract_document_text(path)
        t           = text_to_image(text)
        g, r, d     = self.f.encode(t)
        gc, rc, dc  = g.cpu(), r.cpu(), d.cpu()
        self.ml.update(t)
        meta     = {
            'text_length':  len(text),
            'text_preview': text[:200],
            'format':       path.split('.')[-1]
        }
        checksum = hashlib.md5(open(path,'rb').read()).hexdigest()
        dna      = path + ".dna"
        save_dna(
            dna, gc.squeeze(0), rc.squeeze(0), dc.squeeze(0),
            checksum, 0, None, encrypt, 'document', meta
        )
        dsz = os.path.getsize(dna)
        print(f"  Text:     {len(text):,} chars")
        print(f"  DNA:      {dsz/1024:.1f} KB  "
              f"({(1-dsz/orig)*100:.2f}% smaller)")
        return dna

    def _compress_binary(self, path, encrypt, orig):
        with open(path, 'rb') as f:
            data = f.read()
        size     = 256
        data_arr = np.frombuffer(data[:size*size*3], dtype=np.uint8)
        if len(data_arr) < size*size*3:
            data_arr = np.pad(
                data_arr, (0, size*size*3 - len(data_arr))
            )
        t        = torch.FloatTensor(
            data_arr.reshape(size,size,3)/255.0
        ).permute(2,0,1).unsqueeze(0).to(device)
        g, r, d  = self.f.encode(t)
        gc,rc,dc = g.cpu(), r.cpu(), d.cpu()
        meta     = {
            'original_size': orig,
            'file_type':     path.split('.')[-1]
        }
        checksum = hashlib.md5(data).hexdigest()
        dna      = path + ".dna"
        save_dna(
            dna, gc.squeeze(0), rc.squeeze(0), dc.squeeze(0),
            checksum, 0, None, encrypt, 'binary', meta
        )
        dsz = os.path.getsize(dna)
        print(f"  DNA:      {dsz/1024:.1f} KB  "
              f"({(1-dsz/orig)*100:.2f}% smaller)")
        return dna

    # ── RECONSTRUCT FROM DNA ──────────────────

    def reconstruct(self, dna_path, mode="full", out=None):
        """Auto detect DNA type and reconstruct"""
        if dna_path.endswith('.vdna'):
            return self._reconstruct_video(dna_path, out)
        elif dna_path.endswith('.adna'):
            return self._reconstruct_audio(dna_path, out)
        else:
            return self._reconstruct_image(dna_path, mode, out)

    def _reconstruct_image(self, dna, mode="full", out=None):
        result  = load_dna(dna)
        g, r, d = (result[0].to(device),
                   result[1].to(device),
                   result[2].to(device))
        r0, d0  = torch.zeros_like(r), torch.zeros_like(d)
        cfg     = MODES[mode]

        if cfg["chains"] == "g":
            rec = self.f.decode(g, r0, d0)
        elif cfg["chains"] == "gr":
            rec = self.f.decode(g, r, d0)
        else:
            rec = self.f.decode(g, r, d)

        img = tensor_to_image(rec, cfg["size"])
        if not out:
            out = dna.replace('.dna', f'_{mode}.png')
        img.save(out)
        print(f"  Reconstructed: {out}")
        return out

    def _reconstruct_video(self, vdna, out=None):
        if not MOVIEPY_AVAILABLE:
            print("  Video reconstruction unavailable — "
                  "pip install moviepy==1.0.3")
            return None
        with open(vdna, 'rb') as f:
            f.read(4)
            enc_flag = struct.unpack('B', f.read(1))[0]
            f.read(32)
            raw      = f.read(struct.unpack('I', f.read(4))[0])
        if enc_flag:
            raw = DNA_Enc().dec(raw)
        data        = json.loads(raw.decode())
        meta        = data['meta']
        frames_data = data['frames']
        frames      = []

        for fd in frames_data:
            g   = torch.FloatTensor(fd['g']).unsqueeze(0).to(device)
            r   = torch.FloatTensor(fd['r']).unsqueeze(0).to(device)
            d   = torch.FloatTensor(fd['d']).unsqueeze(0).to(device)
            rec = self.f.decode(g, r, d)
            frames.append(
                (torch.clamp(rec,0,1).squeeze(0)
                 .permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            )

        if not out:
            out = vdna.replace('.vdna', '_reconstructed.mp4')

        # Use saved sample rate to get correct playback speed
        fps         = meta.get('fps', 30)
        sample_rate = meta.get('sample_rate', FRAME_SAMPLE)
        out_fps     = fps / sample_rate

        clip = ImageSequenceClip(frames, fps=out_fps)
        clip.write_videofile(out, verbose=False, logger=None)
        print(f"  Reconstructed: {out}")
        print(f"  Duration:      {len(frames)/out_fps:.1f}s")
        return out

    def _reconstruct_audio(self, adna, out=None):
        if not PYDUB_AVAILABLE:
            print("  Audio reconstruction unavailable — "
                  "install audioop-lts")
            return None
        with open(adna, 'rb') as f:
            f.read(4)
            enc_flag = struct.unpack('B', f.read(1))[0]
            f.read(32)
            raw      = f.read(struct.unpack('I', f.read(4))[0])
        if enc_flag:
            raw = DNA_Enc().dec(raw)
        data   = json.loads(raw.decode())
        meta   = data['meta']
        chunks = []

        for cd in data['chunks']:
            g   = torch.FloatTensor(cd['g']).unsqueeze(0).to(device)
            r   = torch.FloatTensor(cd['r']).unsqueeze(0).to(device)
            d   = torch.FloatTensor(cd['d']).unsqueeze(0).to(device)
            rec = self.f.decode(g, r, d)
            arr = (torch.clamp(rec,0,1).squeeze(0)
                   .permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            # Use mean of all 3 channels — richer reconstruction
            samples = (
                arr.mean(axis=2).flatten().astype(np.float32)
                / 255.0 * 2 - 1
            ) * 32767
            chunk = AudioSegment(
                samples.astype(np.int16).tobytes(),
                frame_rate=meta['sample_rate'],
                sample_width=2,
                channels=1
            )
            chunks.append(chunk)

        if not out:
            out = adna.replace('.adna', '_reconstructed.wav')
        combined = chunks[0]
        for c in chunks[1:]:
            combined += c
        combined.export(out, format='wav')
        print(f"  Reconstructed: {out}")
        print(f"  Duration:      {len(combined)/1000:.1f}s")
        return out

    # ── FULL DEMO — compress + reconstruct + compare ──

    def demo(self, path, encrypt=True):
        """
        Full demo for any file:
        1. Compress to DNA
        2. Reconstruct from DNA
        3. Show 3-stage progression (images only)
        4. Save comparison image
        """
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return

        ftype = detect_file_type(path)
        name  = os.path.splitext(os.path.basename(path))[0]
        orig  = os.path.getsize(path)

        print(f"\n{'='*55}")
        print(f"UNIVERSAL NEURAL FOUNDATION — DEMO")
        print(f"File: {os.path.basename(path)}")
        print(f"Type: {ftype.upper()}")
        print(f"{'='*55}")

        # Step 1 — Compress
        print(f"\nSTEP 1 — COMPRESS")
        start   = time.time()
        dna     = self.compress(path, encrypt=encrypt)
        elapsed = time.time() - start
        if not dna:
            return
        dsz     = os.path.getsize(dna)
        print(f"  Time: {elapsed:.2f}s")

        # Step 2 — Reconstruct
        print(f"\nSTEP 2 — RECONSTRUCT")
        out_path = f"{name}_reconstructed"
        if dna.endswith('.vdna'):
            out_path += '.mp4'
        elif dna.endswith('.adna'):
            out_path += '.wav'
        else:
            out_path += '.png'

        start  = time.time()
        result = self.reconstruct(dna, "full", out_path)
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.2f}s")

        # Step 3 — Progression + Comparison (images only)
        if ftype == 'image':
            print(f"\nSTEP 3 — PROGRESSION & COMPARISON")
            t       = self.img_tensor(path)
            g, r, d = self.f.encode(t)
            r0, d0  = torch.zeros_like(r), torch.zeros_like(d)

            s1  = self.f.decode(g, r0, d0)  # global only
            s2  = self.f.decode(g, r,  d0)  # global + regional
            s3  = self.f.decode(g, r,  d)   # all 3 chains
            rec = self.f.decode(g, r,  d)

            comp_path = f"{name}_comparison.png"
            make_comparison(path, rec, s1, s2, s3, s3, comp_path)
            print(f"  Comparison: {comp_path}")
            print(f"  Columns: Original | "
                  f"Global-32 | Regional-288 | Full-800 | Reconstructed")

        # Summary
        print(f"\n{'='*55}")
        print(f"  Original:  {orig/1024/1024:.3f} MB")
        print(f"  DNA:       {dsz/1024:.2f} KB")
        print(f"  Ratio:     {(1-dsz/orig)*100:.2f}% smaller")
        print(f"  DNA file:  {dna}")
        if result:
            print(f"  Output:    {result}")
        print(f"{'='*55}")

    # ── PROGRESSIVE — Claim 8 ─────────────────

    def progressive(self, image_path, out="progressive_comparison.png"):
        t       = self.img_tensor(image_path)
        g, r, d = self.f.encode(t)
        r0, d0  = torch.zeros_like(r), torch.zeros_like(d)
        s1      = self.f.decode(g, r0, d0)
        s2      = self.f.decode(g, r,  d0)
        s3      = self.f.decode(g, r,  d)
        make_comparison(image_path, s3, s1, s2, s3, s3, out)
        print(f"  Progressive: {out}")
        return out

    # ── BATCH ─────────────────────────────────

    def batch(self, paths, chunk_size=32):
        """
        Batch compress images.
        Processes in chunks to avoid memory issues.
        """
        to, td, out = 0, 0, []
        for chunk_start in range(0, len(paths), chunk_size):
            chunk   = paths[chunk_start:chunk_start+chunk_size]
            tensors = []
            valid   = []
            for p in chunk:
                try:
                    tensors.append(self.img_tensor(p))
                    valid.append(p)
                except:
                    pass
            if not tensors:
                continue
            b = torch.cat(tensors)
            with torch.no_grad():
                gb = self.f.global_encoder(b)
                rb = self.f.regional_encoder(b)
                db = self.f.detail_encoder(b)
            for i, p in enumerate(valid):
                g  = gb[i:i+1].cpu()
                r  = rb[i:i+1].cpu()
                d  = db[i:i+1].cpu()
                cs = hashlib.md5(open(p,'rb').read()).hexdigest()
                dp = p + ".dna"
                save_dna(
                    dp, g.squeeze(0), r.squeeze(0),
                    d.squeeze(0), cs, encrypt=False
                )
                to += os.path.getsize(p)
                td += os.path.getsize(dp)
                out.append(dp)
        if out:
            print(f"  Batch: {len(out)} files | "
                  f"{to/1024/1024:.1f}MB → {td/1024:.1f}KB | "
                  f"{(1-td/to)*100:.1f}% smaller")
        return out

    def federated(self, paths):
        fed = Federated(self.f)
        for p in paths[:3]:
            try: fed.collect(self.img_tensor(p))
            except: pass
        fed.aggregate()

    def stream(self, paths):
        pipe = StreamPipeline(self.f)
        t    = time.time()
        r    = pipe.run(paths)
        el   = time.time() - t
        spd  = len(r)/el if el > 0 else 0
        print(f"  Stream: {len(r)} files in {el:.2f}s "
              f"— {spd:.1f}/s")

    def mobile(self):
        orig = os.path.getsize(WEIGHTS)
        q    = torch.quantization.quantize_dynamic(
            self.f.cpu(), {nn.Linear,nn.Conv2d}, dtype=torch.qint8
        )
        torch.save(q.state_dict(), "foundation_quantized.pth")
        qsz = os.path.getsize("foundation_quantized.pth")
        print(f"  Mobile: {orig/1024/1024:.1f}MB → "
              f"{qsz/1024/1024:.1f}MB  "
              f"({(1-qsz/orig)*100:.1f}% smaller)")
        self.f = self.f.to(device)

    def tiers(self):
        deploy_tiers(WEIGHTS)


# ── RUN ──────────────────────────────────────

if __name__ == "__main__":
    """
    Usage:
      python foundation_v4.py                  — run full demo on demo.png
      python foundation_v4.py photo.jpg        — demo on any image
      python foundation_v4.py video.mp4        — demo on any video
      python foundation_v4.py audio.mp3        — demo on any audio
      python foundation_v4.py document.pdf     — demo on any document
      python foundation_v4.py --all            — run all claims demo
    """
    try:
        s = System()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Get input file from command line or default
    if len(sys.argv) > 1 and sys.argv[1] != '--all':
        input_file = sys.argv[1]
        s.demo(input_file)

    else:
        # Full claims demo on demo.png
        print("\n=== FULL CLAIMS DEMO ===")
        s.demo("demo.png")

        # Find and test any other files in folder
        test_files = [
            f for f in os.listdir('.')
            if (
                any(f.lower().endswith(f'.{ext}')
                    for ext in ['mp4','avi','mp3','wav','pdf','docx'])
                and not f.startswith('demo_')
                and not f.startswith('epigenetic')
                and not f.startswith('progressive')
                and not f.endswith('.dna')
                and not f.endswith('.vdna')
                and not f.endswith('.adna')
            )
        ]
        if test_files:
            print(f"\n=== OTHER FILES FOUND: {len(test_files)} ===")
            for f in test_files[:3]:
                s.demo(f)

        # Epigenetic modes — Claim 10
        if os.path.exists("demo.png.dna"):
            print("\n=== EPIGENETIC MODES (Claim 10) ===")
            dna = "demo.png.dna"
            for mode in ["thumbnail","mobile","full"]:
                out = s.reconstruct(dna, mode,
                                    f"epigenetic_{mode}.png")
                sz  = os.path.getsize(f"epigenetic_{mode}.png")
                print(f"  {mode:10} {sz/1024:.1f}KB — "
                      f"{MODES[mode]['desc']}")
            for f in ["epigenetic_thumbnail.png",
                      "epigenetic_mobile.png"]:
                if os.path.exists(f): os.remove(f)

        # Batch + federated + stream on training images
        training = []
        if os.path.exists("training_images"):
            training = [
                os.path.join("training_images", f)
                for f in os.listdir("training_images")
                if f.endswith(('.jpg','.png'))
            ][:100]  # limit to 100 for demo speed

        if training:
            print("\n=== BATCH (Claim 1) ===")
            start = time.time()
            s.batch(training)
            print(f"  Speed: {len(training)/(time.time()-start):.1f}/s")

            print("\n=== FEDERATED (Claim 4) ===")
            s.federated(training)

            print("\n=== STREAMING ===")
            s.stream(training)

        print("\n=== MOBILE (Claim 3) ===")
        s.mobile()

        print("\n=== THREE TIER (Claim 13) ===")
        s.tiers()

        # Claims summary
        print("\n" + "="*45)
        print("ALL CLAIMS ACTIVE — v5.3")
        print("="*45)
        for num, name in [
            ("1",  "Cross-file shared foundation"),
            ("3",  "Mobile optimization"),
            ("4",  "Federated network"),
            ("6",  "Encryption"),
            ("7",  "Inheritance tree"),
            ("8",  "3D hierarchical DNA"),
            ("9",  "Axon pathways"),
            ("10", "Epigenetic expression"),
            ("11", "Micro-learning"),
            ("12", "Morphogenetic fields"),
            ("13", "Three tier deploy"),
        ]:
            print(f"  Claim {num:<3} {name:<30} ✓")
        print("="*45)
        print("\nFile types:")
        print("  Images:    jpg png bmp webp gif tiff")
        print("  Video:     mp4 avi mov mkv  "
              f"{'✓' if MOVIEPY_AVAILABLE else '✗'}")
        print("  Audio:     mp3 wav flac aac  "
              f"{'✓' if PYDUB_AVAILABLE else '✗'}")
        print("  Documents: pdf docx txt csv  ✓")
        print("  Binary:    any file  ✓")
        print("="*45)