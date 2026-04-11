import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import os
import sys
import struct
import hashlib
import base64
import json
import cv2
import time
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

# ============================================
# DNA VISUALIZER v5.3
# Universal — Images, Video, Audio, Documents
# One foundation — all file types
# Pass any file — auto detect and process
# Inventor: Rohit Kalu Sasane, Pune India 2026
# ============================================

WEIGHTS      = "foundation_v4_weights.pth"
FRAME_SAMPLE = 3
IMAGE_EXT    = {'jpg','jpeg','png','bmp','webp','gif','tiff'}
VIDEO_EXT    = {'mp4','avi','mov','mkv','wmv','flv'}
AUDIO_EXT    = {'mp3','wav','flac','aac','ogg','m4a'}
DOCUMENT_EXT = {'pdf','docx','txt','csv','doc'}


# ── ARCHITECTURE — matches foundation_v4.py ──

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
    def __init__(self):
        super().__init__()
        self.global_encoder    = GlobalEncoder()
        self.regional_encoder  = RegionalEncoder()
        self.detail_encoder    = DetailEncoder()
        self.decoder           = MorphogeneticDecoder()
        self.pathway_classifier = nn.Sequential(
            nn.Linear(800, 256), nn.ReLU(), nn.Linear(256, 5)
        )

    def encode(self, x):
        with torch.no_grad():
            return (self.global_encoder(x),
                    self.regional_encoder(x),
                    self.detail_encoder(x))

    def decode(self, g, r, d):
        with torch.no_grad():
            return self.decoder(g, r, d)


# ── LOAD FOUNDATION ───────────────────────────

def load_foundation():
    if not os.path.exists(WEIGHTS):
        print(f"No weights found: {WEIGHTS}")
        print("Run: python train_foundation.py images")
        sys.exit(1)
    f = Foundation()
    f.load_state_dict(
        torch.load(WEIGHTS, weights_only=True, map_location='cpu')
    )
    f.eval()
    sz = os.path.getsize(WEIGHTS)
    print(f"Foundation: {sz/1024/1024:.1f}MB loaded")
    return f


# ── FILE TYPE DETECTION ───────────────────────

def detect_type(path):
    ext = path.split('.')[-1].lower()
    if ext in IMAGE_EXT:    return 'image'
    if ext in VIDEO_EXT:    return 'video'
    if ext in AUDIO_EXT:    return 'audio'
    if ext in DOCUMENT_EXT: return 'document'
    return 'binary'


# ── DNA STRING ENCODE / DECODE ────────────────

def chains_to_dna(g, r, d):
    """Pack g+r+d into one base64 DNA string"""
    g_arr = g.numpy().flatten().astype(np.float32)
    r_arr = r.numpy().flatten().astype(np.float32)
    d_arr = d.numpy().flatten().astype(np.float32)
    data  = struct.pack('HHH', len(g_arr), len(r_arr), len(d_arr))
    data += g_arr.tobytes() + r_arr.tobytes() + d_arr.tobytes()
    return base64.b64encode(data).decode('utf-8')


def dna_to_chains(dna_string):
    """Unpack DNA string back to g, r, d tensors"""
    data         = base64.b64decode(dna_string)
    g_dim, r_dim, d_dim = struct.unpack('HHH', data[:6])
    off          = 6
    g = np.frombuffer(data[off:off+g_dim*4], dtype=np.float32).copy()
    off += g_dim*4
    r = np.frombuffer(data[off:off+r_dim*4], dtype=np.float32).copy()
    off += r_dim*4
    d = np.frombuffer(data[off:off+d_dim*4], dtype=np.float32).copy()
    return (
        torch.FloatTensor(g).unsqueeze(0),
        torch.FloatTensor(r).unsqueeze(0),
        torch.FloatTensor(d).unsqueeze(0)
    )


# ── TENSOR HELPERS ────────────────────────────

def img_to_tensor(path, size=256):
    return torch.FloatTensor(
        np.array(
            Image.open(path).convert('RGB').resize((size,size))
        )/255.0
    ).permute(2,0,1).unsqueeze(0)


def tensor_to_pil(tensor, size=None):
    arr = (torch.clamp(tensor,0,1).squeeze(0)
           .permute(1,2,0).numpy()*255).astype(np.uint8)
    img = Image.fromarray(arr)
    if size:
        img = img.resize((size,size))
    return img


def frame_to_tensor(frame, size=256):
    img = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ).resize((size,size))
    return torch.FloatTensor(
        np.array(img)/255.0
    ).permute(2,0,1).unsqueeze(0)


def chunk_to_tensor(chunk, size=256):
    if not PYDUB_AVAILABLE:
        return torch.zeros(1,3,size,size)
    samples = np.array(
        chunk.get_array_of_samples()
    ).astype(np.float32)
    if len(samples) == 0:
        samples = np.zeros(size*size)
    samples = samples / (np.max(np.abs(samples)) + 1e-8)
    samples = np.resize(samples, size*size)
    data    = samples.reshape(size, size)
    r_ch    = ((data+1)/2*255).astype(np.uint8)
    g_ch    = (np.abs(data)*255).astype(np.uint8)
    b_ch    = np.clip(
        (1-np.abs(data))*180 + np.abs(np.roll(data,8,1))*75,
        0, 255
    ).astype(np.uint8)
    img = Image.fromarray(np.stack([r_ch,g_ch,b_ch], axis=2))
    return torch.FloatTensor(
        np.array(img)/255.0
    ).permute(2,0,1).unsqueeze(0)


def text_to_tensor(text, size=256):
    chars = np.array(
        [ord(c) for c in text[:size*size]], dtype=np.float32
    )
    if len(chars) < size*size:
        chars = np.pad(chars, (0, size*size-len(chars)))
    chars   = chars / 128.0 - 1.0
    img_arr = (
        (chars[:size*size].reshape(size,size)+1)/2*255
    ).astype(np.uint8)
    img = Image.fromarray(img_arr).convert('RGB')
    return torch.FloatTensor(
        np.array(img)/255.0
    ).permute(2,0,1).unsqueeze(0)


def extract_text(path):
    ext = path.split('.')[-1].lower()
    try:
        if ext == 'pdf':
            import PyPDF2
            reader = PyPDF2.PdfReader(path)
            return ''.join(
                p.extract_text() or '' for p in reader.pages
            )
        elif ext == 'docx':
            import docx as docx_lib
            doc = docx_lib.Document(path)
            return '\n'.join(p.text for p in doc.paragraphs)
        else:
            with open(path,'r',encoding='utf-8',errors='ignore') as f:
                return f.read()
    except Exception as e:
        return f"extraction_failed: {e}"


# ── COMPRESS ANY FILE TO DNA ──────────────────

def file_to_dna(path, foundation):
    """
    Compress any file to DNA string.
    Returns: dna_string, file_type, meta_dict
    """
    ftype    = detect_type(path)
    orig     = os.path.getsize(path)
    checksum = hashlib.md5(open(path,'rb').read()).hexdigest()

    print(f"\n  File:     {os.path.basename(path)}")
    print(f"  Type:     {ftype.upper()}")
    print(f"  Size:     {orig/1024/1024:.3f} MB")

    if ftype == 'image':
        t       = img_to_tensor(path)
        g, r, d = foundation.encode(t)
        recon   = foundation.decode(g, r, d)
        loss    = nn.MSELoss()(recon, t).item()
        dna     = chains_to_dna(g.cpu(), r.cpu(), d.cpu())
        meta    = {
            'file_type': ftype,
            'filename':  os.path.basename(path),
            'orig_size': orig,
            'checksum':  checksum,
            'loss':      loss,
            'chains':    '32+256+512=800'
        }
        print(f"  Loss:     {loss:.6f}")

    elif ftype == 'video':
        cap        = cv2.VideoCapture(path)
        fps        = cap.get(cv2.CAP_PROP_FPS)
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_dnas = []
        frame_idx  = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FRAME_SAMPLE == 0:
                t       = frame_to_tensor(frame)
                g, r, d = foundation.encode(t)
                frame_dnas.append(
                    chains_to_dna(g.cpu(), r.cpu(), d.cpu())
                )
            frame_idx += 1
        cap.release()

        # Pack all frame DNAs as JSON base64
        packed = json.dumps({
            'frames': frame_dnas,
            'fps':    fps,
            'total':  total,
            'sample': FRAME_SAMPLE
        })
        dna  = base64.b64encode(packed.encode()).decode()
        meta = {
            'file_type':  ftype,
            'filename':   os.path.basename(path),
            'orig_size':  orig,
            'checksum':   checksum,
            'fps':        fps,
            'frames':     len(frame_dnas),
            'total':      total
        }
        print(f"  Frames:   {total} → {len(frame_dnas)} compressed")

    elif ftype == 'audio':
        if not PYDUB_AVAILABLE:
            print("  Audio skipped — install audioop-lts")
            return None, ftype, {}
        audio      = AudioSegment.from_file(path)
        chunk_dnas = []
        for i in range(0, len(audio), 1000):
            chunk   = audio[i:i+1000]
            t       = chunk_to_tensor(chunk)
            g, r, d = foundation.encode(t)
            chunk_dnas.append(
                chains_to_dna(g.cpu(), r.cpu(), d.cpu())
            )
        packed = json.dumps({
            'chunks':      chunk_dnas,
            'sample_rate': audio.frame_rate,
            'channels':    audio.channels
        })
        dna  = base64.b64encode(packed.encode()).decode()
        meta = {
            'file_type':   ftype,
            'filename':    os.path.basename(path),
            'orig_size':   orig,
            'checksum':    checksum,
            'sample_rate': audio.frame_rate,
            'chunks':      len(chunk_dnas)
        }
        print(f"  Chunks:   {len(chunk_dnas)} seconds")

    elif ftype == 'document':
        text    = extract_text(path)
        t       = text_to_tensor(text)
        g, r, d = foundation.encode(t)
        recon   = foundation.decode(g, r, d)
        loss    = nn.MSELoss()(recon, t).item()
        dna     = chains_to_dna(g.cpu(), r.cpu(), d.cpu())
        meta    = {
            'file_type':   ftype,
            'filename':    os.path.basename(path),
            'orig_size':   orig,
            'checksum':    checksum,
            'text_length': len(text),
            'preview':     text[:100],
            'loss':        loss
        }
        print(f"  Text:     {len(text):,} chars")

    else:
        # Binary fallback
        with open(path,'rb') as f:
            data = f.read()
        size     = 256
        data_arr = np.frombuffer(data[:size*size*3], dtype=np.uint8)
        if len(data_arr) < size*size*3:
            data_arr = np.pad(
                data_arr, (0, size*size*3-len(data_arr))
            )
        t        = torch.FloatTensor(
            data_arr.reshape(size,size,3)/255.0
        ).permute(2,0,1).unsqueeze(0)
        g, r, d  = foundation.encode(t)
        dna      = chains_to_dna(g.cpu(), r.cpu(), d.cpu())
        meta     = {
            'file_type': ftype,
            'filename':  os.path.basename(path),
            'orig_size': orig,
            'checksum':  checksum
        }

    dna_size = len(dna)
    print(f"  DNA:      {dna_size:,} chars  "
          f"({(1-dna_size/orig)*100:.2f}% smaller)")
    print(f"  Preview:  {dna[:60]}...")

    return dna, ftype, meta


# ── RECONSTRUCT FROM DNA ──────────────────────

def dna_to_file(dna, ftype, meta, foundation,
                out_path=None):
    """
    Reconstruct file from DNA string.
    Auto detect type and reconstruct.
    """
    name = meta.get('filename','output').rsplit('.',1)[0]

    if ftype == 'image':
        g, r, d = dna_to_chains(dna)
        rec     = foundation.decode(g, r, d)
        img     = tensor_to_pil(rec)
        if not out_path:
            out_path = f"{name}_reconstructed.png"
        img.save(out_path)

    elif ftype == 'video':
        if not MOVIEPY_AVAILABLE:
            print("  Video reconstruction unavailable")
            return None
        packed     = json.loads(base64.b64decode(dna).decode())
        fps        = packed['fps']
        sample     = packed.get('sample', FRAME_SAMPLE)
        frame_dnas = packed['frames']
        frames     = []
        for fdna in frame_dnas:
            g, r, d = dna_to_chains(fdna)
            rec     = foundation.decode(g, r, d)
            arr     = (torch.clamp(rec,0,1).squeeze(0)
                       .permute(1,2,0).numpy()*255).astype(np.uint8)
            frames.append(arr)
        if not out_path:
            out_path = f"{name}_reconstructed.mp4"
        clip = ImageSequenceClip(frames, fps=fps/sample)
        clip.write_videofile(out_path, verbose=False, logger=None)

    elif ftype == 'audio':
        if not PYDUB_AVAILABLE:
            print("  Audio reconstruction unavailable")
            return None
        packed = json.loads(base64.b64decode(dna).decode())
        chunks = []
        for cdna in packed['chunks']:
            g, r, d = dna_to_chains(cdna)
            rec     = foundation.decode(g, r, d)
            arr     = (torch.clamp(rec,0,1).squeeze(0)
                       .permute(1,2,0).numpy()*255).astype(np.uint8)
            samples = (
                arr.mean(axis=2).flatten().astype(np.float32)
                / 255.0 * 2 - 1
            ) * 32767
            chunk = AudioSegment(
                samples.astype(np.int16).tobytes(),
                frame_rate=packed['sample_rate'],
                sample_width=2, channels=1
            )
            chunks.append(chunk)
        if not out_path:
            out_path = f"{name}_reconstructed.wav"
        combined = chunks[0]
        for c in chunks[1:]:
            combined += c
        combined.export(out_path, format='wav')

    else:
        # Image or document reconstruction
        g, r, d = dna_to_chains(dna)
        rec     = foundation.decode(g, r, d)
        img     = tensor_to_pil(rec)
        if not out_path:
            out_path = f"{name}_reconstructed.png"
        img.save(out_path)

    print(f"  Reconstructed: {out_path}")
    return out_path


# ── PROGRESSION CANVAS ────────────────────────

def make_progression(path, foundation, ftype, out=None):
    """
    3 stage progression for images only.
    Shows how DNA chains build up quality.
    """
    if ftype != 'image':
        print(f"  Progression only available for images.")
        return None

    t       = img_to_tensor(path)
    g, r, d = foundation.encode(t)
    r0, d0  = torch.zeros_like(r), torch.zeros_like(d)

    s1  = foundation.decode(g, r0, d0)  # global only
    s2  = foundation.decode(g, r,  d0)  # global + regional
    s3  = foundation.decode(g, r,  d)   # all 3

    orig = Image.open(path).convert('RGB').resize((256,256))
    i1   = tensor_to_pil(s1, 256)
    i2   = tensor_to_pil(s2, 256)
    i3   = tensor_to_pil(s3, 256)
    irec = tensor_to_pil(s3, 256)

    pad   = 10
    lh    = 28
    w     = 5*256 + 6*pad
    h     = 256 + lh + pad*2
    canvas = Image.new('RGB', (w, h), (15,15,25))
    draw   = ImageDraw.Draw(canvas)

    items = [
        (orig, "Original",       (255,255,255)),
        (i1,   "Global  (32)",   (150,150,255)),
        (i2,   "Regional (288)", (100,200,255)),
        (i3,   "Full (800)",     (100,255,150)),
        (irec, "Reconstructed",  (80, 255, 80)),
    ]
    for i, (img, label, color) in enumerate(items):
        x = pad + i*(256+pad)
        canvas.paste(img, (x, lh+pad))
        draw.text((x+128, 5), label,
                  fill=color, anchor="mt")

    draw.text(
        (w//2, h-6),
        "Universal Neural Foundation — Rohit Sasane 2026",
        fill=(70,70,70), anchor="mb"
    )

    if not out:
        name = os.path.splitext(os.path.basename(path))[0]
        out  = f"{name}_progression.png"
    canvas.save(out)
    print(f"  Progression: {out}")
    return out


# ── PITCH CARD ────────────────────────────────

def make_pitch_card(path, dna, meta, recon_path,
                    ftype, foundation):
    """
    Professional pitch card for any file type.
    Shows original info + DNA + reconstruction stats.
    """
    orig_size = meta.get('orig_size', 0)
    dna_size  = len(dna)
    ratio     = (1 - dna_size/orig_size)*100 if orig_size > 0 else 0
    name      = os.path.splitext(os.path.basename(path))[0]

    w, h = 1200, 700
    card = Image.new('RGB', (w, h), (12, 12, 20))
    draw = ImageDraw.Draw(card)

    # Header
    draw.rectangle([(0,0),(w,80)], fill=(20,20,35))
    draw.text((w//2, 18),
              "UNIVERSAL NEURAL FOUNDATION v5.3",
              fill=(100,200,255), anchor="mt")
    draw.text((w//2, 45),
              "Invented by Rohit Kalu Sasane — Pune, India — 2026",
              fill=(140,140,140), anchor="mt")
    draw.text((w//2, 65),
              "Patent Pending — 13 Claims Filed",
              fill=(100,180,100), anchor="mt")

    # Show images for image type
    if ftype == 'image' and os.path.exists(recon_path):
        orig_img  = Image.open(path).convert('RGB').resize((400,400))
        recon_img = Image.open(recon_path).convert('RGB').resize((400,400))
        card.paste(orig_img,  (60,  110))
        card.paste(recon_img, (740, 110))
        draw.text((260, 92),
                  f"ORIGINAL — {orig_size/1024/1024:.2f} MB",
                  fill=(255,255,255), anchor="mt")
        draw.text((940, 92),
                  f"RECONSTRUCTED — {dna_size:,} chars",
                  fill=(100,255,100), anchor="mt")
        draw.text((w//2, 310), "→",
                  fill=(255,200,0), anchor="mm")
        draw.text((w//2, 350), "DNA",
                  fill=(255,200,0), anchor="mm")
        draw.text((w//2, 385),
                  dna[:50]+"...",
                  fill=(180,180,80), anchor="mm")

    else:
        # Non-image file type card
        draw.rectangle([(100,110),(1100,550)],
                       fill=(20,20,35),
                       outline=(40,40,60), width=2)
        draw.text((w//2, 160),
                  f"FILE TYPE: {ftype.upper()}",
                  fill=(100,200,255), anchor="mt")
        draw.text((w//2, 210),
                  f"Original: {os.path.basename(path)}",
                  fill=(200,200,200), anchor="mt")
        draw.text((w//2, 250),
                  f"Size: {orig_size/1024/1024:.3f} MB → "
                  f"{dna_size:,} characters",
                  fill=(100,255,100), anchor="mt")
        draw.text((w//2, 300),
                  f"Compression: {ratio:.2f}% smaller",
                  fill=(255,200,0), anchor="mt")
        draw.text((w//2, 360),
                  "DNA PREVIEW:",
                  fill=(150,150,150), anchor="mt")
        draw.text((w//2, 395),
                  dna[:80]+"...",
                  fill=(180,180,80), anchor="mt")
        if ftype == 'video':
            draw.text((w//2, 450),
                      f"Frames compressed: {meta.get('frames',0)}",
                      fill=(200,200,200), anchor="mt")
        elif ftype == 'audio':
            draw.text((w//2, 450),
                      f"Audio chunks: {meta.get('chunks',0)} seconds",
                      fill=(200,200,200), anchor="mt")
        elif ftype == 'document':
            draw.text((w//2, 450),
                      f"Text chars: {meta.get('text_length',0):,}",
                      fill=(200,200,200), anchor="mt")

    # Footer stats
    draw.rectangle([(0,620),(w,700)], fill=(18,18,30))
    draw.text((w//2, 635),
              f"Original: {orig_size/1024/1024:.3f} MB  →  "
              f"DNA: {dna_size:,} chars  →  "
              f"{ratio:.2f}% smaller  →  "
              f"Type: {ftype.upper()}",
              fill=(100,255,100), anchor="mt")
    draw.text((w//2, 665),
              "One Foundation — All File Types — "
              "Cross-file Inheritance — 13 Patent Claims",
              fill=(100,100,100), anchor="mt")

    out = f"{name}_pitch_card.png"
    card.save(out)
    print(f"  Pitch card: {out}")
    return out


# ── SAVE DNA FILE ─────────────────────────────

def save_dna_file(path, dna, ftype, meta):
    """Save DNA to readable text file"""
    name    = os.path.splitext(os.path.basename(path))[0]
    out     = f"{name}.dna.txt"
    orig    = meta.get('orig_size', 0)
    dna_sz  = len(dna)
    ratio   = (1 - dna_sz/orig)*100 if orig > 0 else 0

    with open(out, 'w') as f:
        f.write("="*60 + "\n")
        f.write("UNIVERSAL NEURAL FOUNDATION v5.3 — DNA FILE\n")
        f.write("Invented by Rohit Kalu Sasane, Pune India 2026\n")
        f.write("="*60 + "\n\n")
        f.write(f"File:         {meta.get('filename','')}\n")
        f.write(f"Type:         {ftype.upper()}\n")
        f.write(f"Original:     {orig:,} bytes "
                f"({orig/1024/1024:.3f} MB)\n")
        f.write(f"DNA size:     {dna_sz:,} characters\n")
        f.write(f"Compression:  {ratio:.2f}% smaller\n")
        f.write(f"Checksum:     {meta.get('checksum','')}\n")
        if 'loss' in meta:
            f.write(f"Loss:         {meta['loss']:.6f}\n")
        if ftype == 'video':
            f.write(f"Frames:       {meta.get('frames',0)} "
                    f"of {meta.get('total',0)}\n")
        elif ftype == 'audio':
            f.write(f"Chunks:       {meta.get('chunks',0)} seconds\n")
        elif ftype == 'document':
            f.write(f"Text chars:   {meta.get('text_length',0):,}\n")
            f.write(f"Preview:      {meta.get('preview','')[:80]}\n")
        f.write(f"\nDNA STRING:\n")
        f.write("-"*60 + "\n")
        f.write(dna + "\n")
        f.write("-"*60 + "\n")
        f.write("\nTo reconstruct: python dna_visualizer.py\n")
        f.write("Choose option 2 and paste this DNA string.\n")

    print(f"  DNA file:   {out}")
    return out


# ── FULL DEMO ─────────────────────────────────

def full_demo(path):
    """
    Complete demo for any file type:
    1. Load foundation
    2. Compress to DNA
    3. Save DNA file
    4. Reconstruct from DNA
    5. Progression (images only)
    6. Pitch card
    """
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    ftype = detect_type(path)
    name  = os.path.splitext(os.path.basename(path))[0]

    print("\n" + "="*55)
    print("UNIVERSAL NEURAL FOUNDATION — FULL DEMO")
    print(f"File: {os.path.basename(path)}")
    print(f"Type: {ftype.upper()}")
    print("="*55)

    # Load foundation
    foundation = load_foundation()

    # Step 1 — Compress
    print("\nSTEP 1 — COMPRESS TO DNA")
    start       = time.time()
    dna, ftype, meta = file_to_dna(path, foundation)
    if dna is None:
        return
    print(f"  Time: {time.time()-start:.2f}s")

    # Step 2 — Save DNA file
    print("\nSTEP 2 — SAVE DNA FILE")
    dna_file = save_dna_file(path, dna, ftype, meta)

    # Step 3 — Reconstruct
    print("\nSTEP 3 — RECONSTRUCT FROM DNA")
    start     = time.time()
    recon     = dna_to_file(dna, ftype, meta, foundation)
    print(f"  Time: {time.time()-start:.2f}s")

    # Step 4 — Progression (images only)
    if ftype == 'image':
        print("\nSTEP 4 — 3 STAGE PROGRESSION")
        make_progression(path, foundation, ftype)

    # Step 5 — Pitch card
    print("\nSTEP 5 — PITCH CARD")
    make_pitch_card(path, dna, meta, recon or '',
                    ftype, foundation)

    # Summary
    orig    = meta.get('orig_size', 0)
    dna_sz  = len(dna)
    ratio   = (1 - dna_sz/orig)*100 if orig > 0 else 0
    print("\n" + "="*55)
    print("DEMO COMPLETE")
    print(f"  Original:  {orig/1024/1024:.3f} MB")
    print(f"  DNA:       {dna_sz:,} chars")
    print(f"  Ratio:     {ratio:.2f}% smaller")
    print(f"  DNA file:  {dna_file}")
    if recon:
        print(f"  Output:    {recon}")
    print("="*55)


# ── MAIN ─────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("UNIVERSAL NEURAL FOUNDATION v5.3")
    print("DNA VISUALIZER — All File Types")
    print("Inventor: Rohit Kalu Sasane, Pune India 2026")
    print("="*55)

    # Command line: python dna_visualizer.py photo.jpg
    if len(sys.argv) > 1:
        full_demo(sys.argv[1])
        sys.exit(0)

    print("\n1. Full demo    — compress + reconstruct + pitch card")
    print("2. Compress     — file → DNA string")
    print("3. Reconstruct  — DNA string → file")
    print("4. Progression  — show 3 stage quality (images)")
    print()

    choice = input("Enter 1/2/3/4: ").strip()

    if choice == '1':
        path = input("File path: ").strip()
        full_demo(path)

    elif choice == '2':
        path = input("File path: ").strip()
        if not os.path.exists(path):
            print("File not found.")
        else:
            foundation = load_foundation()
            ftype      = detect_type(path)
            dna, ftype, meta = file_to_dna(path, foundation)
            if dna:
                save_dna_file(path, dna, ftype, meta)

    elif choice == '3':
        print("Paste DNA string:")
        dna   = input().strip()
        ftype = input("File type (image/video/audio/document): ").strip()
        meta  = {'filename': 'output', 'orig_size': 1}
        foundation = load_foundation()
        dna_to_file(dna, ftype, meta, foundation)

    elif choice == '4':
        path = input("Image path: ").strip()
        if not os.path.exists(path):
            print("File not found.")
        else:
            foundation = load_foundation()
            ftype      = detect_type(path)
            make_progression(path, foundation, ftype)

    else:
        print("Invalid choice.")