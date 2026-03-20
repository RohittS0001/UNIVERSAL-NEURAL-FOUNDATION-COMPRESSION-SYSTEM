import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import os
import requests
import tempfile

# ============================================
# UNIVERSAL FOUNDATION TRAINER v5.2
# Trains on ALL file types:
# Images, Video frames, Audio spectrograms,
# Documents — 1000+ per type
# Inventor: Rohit Kalu Sasane, Pune India 2026
# ============================================

SAVE_PATH = "foundation_v4_weights.pth"
IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── ARCHITECTURE — must match foundation_v4.py ─

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
        self.field_receiver = nn.Linear(64, 512*8*8)
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
        self.global_encoder = GlobalEncoder()
        self.regional_encoder = RegionalEncoder()
        self.detail_encoder = DetailEncoder()
        self.decoder = MorphogeneticDecoder()
        self.pathway_classifier = nn.Sequential(
            nn.Linear(800, 256), nn.ReLU(), nn.Linear(256, 5)
        )
    def forward(self, x):
        g = self.global_encoder(x)
        r = self.regional_encoder(x)
        d = self.detail_encoder(x)
        return self.decoder(g, r, d), g, r, d


# ── DATA GENERATORS ──────────────────────────

def download_images(target=1000):
    """Download diverse images from Unsplash"""
    os.makedirs("training_images", exist_ok=True)
    categories = [
        'nature','city','people','animals','food',
        'architecture','technology','sports','travel',
        'abstract','flowers','ocean','mountains','night',
        'portrait','street','business','space','art'
    ]
    count = len([f for f in os.listdir("training_images")
                 if f.endswith('.jpg')])
    if count >= target:
        print(f"  Images: {count} already downloaded")
        return
    print(f"  Downloading images (target: {target})...")
    per_cat = target // len(categories) + 1
    for cat in categories:
        for i in range(per_cat):
            if count >= target:
                break
            try:
                url = f"https://source.unsplash.com/256x256/?{cat}&sig={count}"
                r = requests.get(url, timeout=8)
                if r.status_code == 200:
                    with open(f"training_images/{cat}_{i:04d}.jpg",'wb') as f:
                        f.write(r.content)
                    count += 1
                    if count % 100 == 0:
                        print(f"    {count}/{target} images")
            except:
                pass
    print(f"  Images ready: {count}")


def generate_video_frames(target=1000):
    """Generate synthetic video frame sequences"""
    folder = "training_data/video_frames"
    os.makedirs(folder, exist_ok=True)
    count = len([f for f in os.listdir(folder)
                 if f.endswith('.jpg')])
    if count >= target:
        print(f"  Video frames: {count} already generated")
        return
    print(f"  Generating video frames (target: {target})...")
    sequences = target // 20 + 1
    count = 0
    for seq in range(sequences):
        if count >= target:
            break
        # Random motion parameters
        r_speed = np.random.randint(1, 20)
        g_speed = np.random.randint(1, 15)
        hue = np.random.randint(0, 255)
        for frame in range(20):
            if count >= target:
                break
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            for y in range(256):
                for x in range(256):
                    img[y,x,0] = (x*r_speed + frame*10 + seq*30) % 256
                    img[y,x,1] = (y*g_speed + frame*5) % 256
                    img[y,x,2] = (hue + frame*12) % 256
            Image.fromarray(img).save(
                f"{folder}/seq{seq:04d}_f{frame:03d}.jpg"
            )
            count += 1
    print(f"  Video frames ready: {count}")


def generate_spectrograms(target=1000):
    """Generate audio spectrogram images"""
    folder = "training_data/spectrograms"
    os.makedirs(folder, exist_ok=True)
    count = len([f for f in os.listdir(folder)
                 if f.endswith('.jpg')])
    if count >= target:
        print(f"  Spectrograms: {count} already generated")
        return
    print(f"  Generating spectrograms (target: {target})...")
    for i in range(target):
        # Multi-frequency signal simulation
        freqs = np.random.uniform(0.5, 20, size=5)
        amps = np.random.uniform(0.1, 1.0, size=5)
        t = np.linspace(0, 1, 256*256)
        signal = sum(a * np.sin(2*np.pi*f*t)
                     for f, a in zip(freqs, amps))
        signal += 0.1 * np.random.randn(len(signal))
        signal = signal / (np.max(np.abs(signal)) + 1e-8)
        # Convert to RGB spectrogram
        data = signal.reshape(256, 256)
        r = ((data + 1) / 2 * 255).astype(np.uint8)
        g = (np.abs(data) * 255).astype(np.uint8)
        b = ((1 - np.abs(data)) * 255).astype(np.uint8)
        img = Image.fromarray(
            np.stack([r, g, b], axis=2)
        )
        img.save(f"{folder}/spec_{i:04d}.jpg")
        if (i+1) % 200 == 0:
            print(f"    {i+1}/{target} spectrograms")
    print(f"  Spectrograms ready: {target}")


def generate_documents(target=1000):
    """Generate synthetic document page images"""
    folder = "training_data/documents"
    os.makedirs(folder, exist_ok=True)
    count = len([f for f in os.listdir(folder)
                 if f.endswith('.jpg')])
    if count >= target:
        print(f"  Documents: {count} already generated")
        return
    print(f"  Generating document images (target: {target})...")
    for i in range(target):
        bg = np.random.randint(240, 256)
        img = Image.new('RGB', (256,256),
                        color=(bg, bg, bg))
        draw = ImageDraw.Draw(img)
        # Title
        tw = np.random.randint(80, 220)
        draw.rectangle([20, 15, tw, 28],
                       fill=(np.random.randint(0,50),)*3)
        # Text lines
        for line in range(np.random.randint(8, 18)):
            y = 40 + line * 14
            lw = np.random.randint(40, 230)
            gray = np.random.randint(30, 100)
            draw.rectangle([20, y, lw, y+7],
                           fill=(gray, gray, gray))
        # Sometimes add a box (table/image)
        if np.random.random() > 0.5:
            x1 = np.random.randint(20, 100)
            y1 = np.random.randint(100, 150)
            x2 = x1 + np.random.randint(60, 140)
            y2 = y1 + np.random.randint(40, 80)
            gray = np.random.randint(150, 220)
            draw.rectangle([x1,y1,x2,y2],
                           fill=(gray,gray,gray))
        img.save(f"{folder}/doc_{i:04d}.jpg")
        if (i+1) % 200 == 0:
            print(f"    {i+1}/{target} documents")
    print(f"  Documents ready: {target}")


# ── DATA LOADER ───────────────────────────────

def load_all_training_data(max_per_type=250):
    """Load images from all training folders"""
    folders = [
        ("training_images",             "Images"),
        ("training_data/video_frames",  "Video"),
        ("training_data/spectrograms",  "Audio"),
        ("training_data/documents",     "Docs"),
    ]
    all_tensors = []
    for folder, name in folders:
        if not os.path.exists(folder):
            continue
        count = 0
        files = [f for f in sorted(os.listdir(folder))
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]
        np.random.shuffle(files)
        for fname in files[:max_per_type]:
            try:
                img = Image.open(
                    os.path.join(folder, fname)
                ).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
                t = torch.FloatTensor(
                    np.array(img)/255.0
                ).permute(2,0,1).unsqueeze(0)
                all_tensors.append(t)
                count += 1
            except:
                pass
        print(f"  {name:8}: {count} samples loaded")
    return all_tensors


# ── TRAINING LOOP ─────────────────────────────

def train(epochs=500, batch_size=16):
    print("\n" + "="*55)
    print("UNIVERSAL FOUNDATION TRAINER v5.2")
    print("Training on: Images + Video + Audio + Documents")
    print(f"Device: {DEVICE}")
    print("Inventor: Rohit Kalu Sasane, Pune India 2026")
    print("="*55)

    # Generate all training data
    print("\nPreparing training data...")
    download_images(target=1000)
    generate_video_frames(target=1000)
    generate_spectrograms(target=1000)
    generate_documents(target=1000)

    # Load data
    print("\nLoading training data...")
    tensors = load_all_training_data(max_per_type=250)
    total = len(tensors)
    print(f"\nTotal training samples: {total}")

    if total == 0:
        print("No training data found.")
        return

    # Initialize foundation
    foundation = Foundation().to(DEVICE)

    # Load existing weights if available
    if os.path.exists(SAVE_PATH):
        try:
            foundation.load_state_dict(
                torch.load(SAVE_PATH, map_location=DEVICE,
                          weights_only=True)
            )
            print("Loaded existing weights — continuing training")
        except:
            print("Starting fresh training")

    optimizer = torch.optim.Adam(
        foundation.parameters(), lr=0.001
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=200, gamma=0.5
    )
    loss_fn = nn.MSELoss()
    best_loss = float('inf')

    print(f"\nTraining {total} samples")
    print(f"Epochs: {epochs}  Batch: {batch_size}")
    print(f"Expected time on GPU: 30-60 minutes")
    print(f"Expected time on CPU: several hours\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        np.random.shuffle(tensors)
        batches = 0

        for i in range(0, total, batch_size):
            batch_list = tensors[i:i+batch_size]
            if not batch_list:
                continue
            batch = torch.cat(batch_list).to(DEVICE)
            optimizer.zero_grad()
            out, g, r, d = foundation(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        scheduler.step()
        avg = epoch_loss / max(batches, 1)

        if avg < best_loss:
            best_loss = avg
            # Atomic save
            tmp = SAVE_PATH + ".tmp"
            torch.save(foundation.state_dict(), tmp)
            if os.path.exists(SAVE_PATH):
                os.remove(SAVE_PATH)
            os.rename(tmp, SAVE_PATH)

        if (epoch+1) % 50 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:>4}/{epochs} — "
                  f"Loss: {avg:.6f}  "
                  f"Best: {best_loss:.6f}  "
                  f"LR: {lr:.6f}")

    sz = os.path.getsize(SAVE_PATH)
    print(f"\n{'='*55}")
    print(f"Training complete!")
    print(f"Best loss:    {best_loss:.6f}")
    print(f"Saved:        {SAVE_PATH} ({sz/1024/1024:.1f}MB)")
    print(f"Trained on:   {total} samples")
    print(f"File types:   Images + Video + Audio + Documents")
    print(f"Now run:      python foundation_v4.py")
    print(f"{'='*55}")


if __name__ == "__main__":
    if os.path.exists(SAVE_PATH):
        print(f"Weights exist: {SAVE_PATH}")
        print("Delete to retrain from scratch.")
        print("Continuing with data generation only...\n")
        print("Preparing training data...")
        download_images(target=1000)
        generate_video_frames(target=1000)
        generate_spectrograms(target=1000)
        generate_documents(target=1000)
        print("\nAll data ready. Delete weights to retrain.")
    else:
        train(epochs=500, batch_size=16)
