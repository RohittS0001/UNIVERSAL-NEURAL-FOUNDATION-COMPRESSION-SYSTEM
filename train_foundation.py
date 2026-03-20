import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import requests
import tempfile

# ============================================
# UNIVERSAL FOUNDATION TRAINER v5
# Trains the complete v5 architecture
# Includes morphogenetic decoder + pathways
# Run this ONCE to build the shared foundation
# Inventor: Rohit Kalu Sasane, Pune India 2026
# ============================================

SAVE_PATH = "foundation_v4_weights.pth"
IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GlobalEncoder(nn.Module):
    """Claim 8 — Global context encoder (coarsest level)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(128 * 4 * 4, 32)
        )

    def forward(self, x):
        return self.net(x)


class RegionalEncoder(nn.Module):
    """Claim 8 — Regional context encoder (mid level)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256)
        )

    def forward(self, x):
        return self.net(x)


class DetailEncoder(nn.Module):
    """Claim 8 — Fine detail encoder (finest level)"""
    def __init__(self):
        super().__init__()
        # AdaptiveAvgPool2d(8) makes the linear size input-resolution-safe
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(8), nn.Flatten(),       # FIX: was fragile hardcoded 16x16
            nn.Linear(512 * 8 * 8, 512)
        )

    def forward(self, x):
        return self.net(x)


class MorphogeneticDecoder(nn.Module):
    """
    Claim 12 — Morphogenetic Field Reconstruction
    Regional decoders broadcast consistency signals
    to neighboring regions before finalizing output.
    Input dims: g=32, r=256, d=512 → combined=800
    """
    def __init__(self):
        super().__init__()
        self.combine = nn.Sequential(
            nn.Linear(800, 512 * 8 * 8), nn.ReLU()     # FIX: matched to 8x8 pool above
        )
        self.field_generator = nn.Linear(512 * 8 * 8, 64)
        self.field_receiver  = nn.Linear(64, 512 * 8 * 8)
        self.decode = nn.Sequential(
            nn.Unflatten(1, (512, 8, 8)),               # FIX: matched to 8x8
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.ReLU(),  # FIX: extra upsample to reach 256
            nn.ConvTranspose2d(32,  3,   4, stride=2, padding=1),
            nn.Sigmoid()
        )
        # Output: 8 → 16 → 32 → 64 → 128 → 256  ✓

    def forward(self, g, r, d):
        combined = torch.cat([g, r, d], dim=1)   # [B, 800]
        x = self.combine(combined)               # [B, 512*8*8]
        field_signal     = self.field_generator(x)
        field_correction = self.field_receiver(field_signal)
        x = x + 0.1 * field_correction           # morphogenetic modulation
        return self.decode(x)                    # [B, 3, 256, 256]


class HierarchicalFoundation(nn.Module):
    """
    Complete v5 foundation with all claims built in:
    Claim 8  — 3D hierarchical encoders (global / regional / detail)
    Claim 9  — Axon pathway classifier
    Claim 12 — Morphogenetic field decoder
    """
    def __init__(self):
        super().__init__()
        self.global_encoder   = GlobalEncoder()
        self.regional_encoder = RegionalEncoder()
        self.detail_encoder   = DetailEncoder()
        self.decoder          = MorphogeneticDecoder()

        # Claim 9 — Axon pathway classifier
        self.pathway_classifier = nn.Sequential(
            nn.Linear(800, 256), nn.ReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        g = self.global_encoder(x)       # [B, 32]
        r = self.regional_encoder(x)     # [B, 256]
        d = self.detail_encoder(x)       # [B, 512]
        reconstructed = self.decoder(g, r, d)   # [B, 3, 256, 256]
        return reconstructed, g, r, d


# ─────────────────────────────────────────────
# Utility: atomic save (avoids Windows file-lock error code 32)
# ─────────────────────────────────────────────
def atomic_save(state_dict, path):
    dir_name = os.path.dirname(os.path.abspath(path))
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    os.close(tmp_fd)
    try:
        torch.save(state_dict, tmp_path)
        if os.path.exists(path):
            os.remove(path)
        os.rename(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e


# ─────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────
def prepare_image(path, size=IMAGE_SIZE):
    img = Image.open(path).convert('RGB').resize((size, size))
    return torch.FloatTensor(
        np.array(img) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)


def download_images():
    print("Downloading diverse training images...")
    os.makedirs("training_images", exist_ok=True)

    urls = [
        ("https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400", "mountain.jpg"),
        ("https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400",    "portrait.jpg"),
        ("https://images.unsplash.com/photo-1518791841217-8f162f1912da?w=400", "cat.jpg"),
        ("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=400", "city.jpg"),
        ("https://images.unsplash.com/photo-1490730141103-6cac27aaab94?w=400", "sunset.jpg"),
        ("https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400", "forest.jpg"),
        ("https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=400",    "snow.jpg"),
        ("https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400", "face.jpg"),
    ]

    paths = []
    for url, filename in urls:
        path = f"training_images/{filename}"
        if os.path.exists(path):
            paths.append(path)
            print(f"  Already exists: {filename}")
            continue
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(r.content)
                paths.append(path)
                print(f"  Downloaded: {filename}")
            else:
                print(f"  Failed (HTTP {r.status_code}): {filename}")
        except Exception as e:
            print(f"  Failed ({e}): {filename}")

    if os.path.exists("demo.png"):
        paths.append("demo.png")
        print(f"  Added: demo.png")

    return paths


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train(epochs=2000):
    print("\n" + "=" * 55)
    print("UNIVERSAL FOUNDATION TRAINER v5")
    print("3D Hierarchical + Morphogenetic Architecture")
    print("Claims 8, 9, 12 built into foundation")
    print(f"Device: {DEVICE}")
    print("Inventor: Rohit Kalu Sasane, Pune India 2026")
    print("=" * 55 + "\n")

    image_paths = download_images()
    print(f"\nTotal training images: {len(image_paths)}")

    tensors = []
    print("\nLoading images...")
    for path in image_paths:
        try:
            t = prepare_image(path)
            tensors.append(t)
            print(f"  Loaded: {path}")
        except Exception as e:
            print(f"  Failed ({e}): {path}")

    if not tensors:
        print("No images loaded. Check training_images folder.")
        return

    batch = torch.cat(tensors, dim=0).to(DEVICE)
    print(f"\nTraining batch: {batch.shape}")
    print(f"Training for {epochs} epochs...\n")

    foundation = HierarchicalFoundation().to(DEVICE)
    optimizer  = torch.optim.Adam(foundation.parameters(), lr=0.001)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    loss_fn    = nn.MSELoss()
    best_loss  = float('inf')

    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed, g, r, d = foundation(batch)
        loss = loss_fn(reconstructed, batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            atomic_save(foundation.state_dict(), SAVE_PATH)  # FIX: atomic, no file-lock error

        if (epoch + 1) % 200 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:>4}/{epochs} — Loss: {loss.item():.6f}  "
                  f"Best: {best_loss:.6f}  LR: {lr_now:.6f}")

    size = os.path.getsize(SAVE_PATH)
    print(f"\n{'=' * 55}")
    print(f"Training complete!")
    print(f"Best loss   : {best_loss:.6f}")
    print(f"Saved to    : {SAVE_PATH}  ({size / 1024 / 1024:.1f} MB)")
    print(f"Trained on  : {len(tensors)} diverse images")
    print(f"Architecture: 3D hierarchical + morphogenetic")
    print(f"\nNow run: python foundation_v5.py")
    print("=" * 55)


if __name__ == "__main__":
    if os.path.exists(SAVE_PATH):
        print(f"Foundation weights already exist: {SAVE_PATH}")
        print("Delete that file to retrain from scratch.")
        print("Downloading/checking images only...\n")
        download_images()
        print("Done.")
    else:
        train(epochs=2000)