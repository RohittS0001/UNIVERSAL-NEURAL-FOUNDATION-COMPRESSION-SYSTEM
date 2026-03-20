import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import os
import tempfile

# ============================================
# UNIVERSAL FOUNDATION TRAINER v5.2 FINAL
# Trains on ALL file types:
# Images (CIFAR10+CIFAR100) — 5000 real photos
# Video frames — 5000 motion sequences
# Audio spectrograms — 5000 sound patterns
# Documents — 5000 page layouts
# Total: 20000 diverse samples
# Device: auto GPU/CPU
# Inventor: Rohit Kalu Sasane, Pune India 2026
# ============================================

SAVE_PATH  = "foundation_v4_weights.pth"
IMAGE_SIZE = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── ARCHITECTURE — must match foundation_v4.py ──

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
    """Claim 12 — Morphogenetic Field Reconstruction"""
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
    """Claims 8 9 12 — full architecture"""
    def __init__(self):
        super().__init__()
        self.global_encoder   = GlobalEncoder()
        self.regional_encoder = RegionalEncoder()
        self.detail_encoder   = DetailEncoder()
        self.decoder          = MorphogeneticDecoder()
        self.pathway_classifier = nn.Sequential(
            nn.Linear(800, 256), nn.ReLU(), nn.Linear(256, 5)
        )

    def forward(self, x):
        g = self.global_encoder(x)
        r = self.regional_encoder(x)
        d = self.detail_encoder(x)
        return self.decoder(g, r, d), g, r, d


# ── ATOMIC SAVE — prevents file corruption ───

def atomic_save(state_dict, path):
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    os.close(fd)
    try:
        torch.save(state_dict, tmp)
        if os.path.exists(path):
            os.remove(path)
        os.rename(tmp, path)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise e


# ── DATA GENERATORS ──────────────────────────

def download_images_cifar(target=5000):
    """Download real diverse images using CIFAR10 + CIFAR100"""
    os.makedirs("training_images", exist_ok=True)
    existing = len([f for f in os.listdir("training_images")
                    if f.endswith('.jpg')])
    if existing >= target:
        print(f"  Images: {existing} already ready")
        return

    print(f"  Downloading CIFAR images (target: {target})...")
    try:
        import torchvision.datasets as datasets
        count = existing

        for ds_class, root in [
            (datasets.CIFAR10,  './cifar10'),
            (datasets.CIFAR100, './cifar100')
        ]:
            if count >= target:
                break
            ds = ds_class(root=root, download=True, train=True)
            for i in range(len(ds)):
                if count >= target:
                    break
                img, _ = ds[i]
                img.resize((256, 256)).save(
                    f"training_images/img_{count:05d}.jpg"
                )
                count += 1
                if count % 500 == 0:
                    print(f"    {count}/{target} images")

        print(f"  Images ready: {count}")
    except Exception as e:
        print(f"  CIFAR download failed: {e}")
        print(f"  Using existing {existing} images")


def generate_video_frames(target=5000):
    """Generate diverse synthetic video frame sequences"""
    folder = "training_data/video_frames"
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder)
                    if f.endswith('.jpg')])
    if existing >= target:
        print(f"  Video frames: {existing} already ready")
        return

    print(f"  Generating video frames (target: {target})...")
    count    = existing
    seq_start = existing // 20

    for seq in range(seq_start, seq_start + 500):
        if count >= target:
            break
        r_speed = np.random.randint(1, 20)
        g_speed = np.random.randint(1, 15)
        hue     = np.random.randint(0, 255)
        pattern = np.random.randint(0, 5)

        for frame in range(20):
            if count >= target:
                break
            img = np.zeros((256, 256, 3), dtype=np.uint8)

            for y in range(256):
                for x in range(256):
                    if pattern == 0:   # Moving gradient
                        img[y,x] = [
                            (x*r_speed + frame*10 + seq*5) % 256,
                            (y*g_speed + frame*5) % 256,
                            (hue + frame*12) % 256
                        ]
                    elif pattern == 1: # Radial pulse
                        cx = 128 + frame*2
                        dist = int(((x-cx)**2+(y-128)**2)**0.5) % 256
                        img[y,x] = [dist, (dist+hue)%256, 255-dist]
                    elif pattern == 2: # Wave
                        val = int((np.sin(x/20+frame*0.3)+1)*127)
                        img[y,x] = [val, (val+100)%256, hue]
                    elif pattern == 3: # Checkerboard
                        c = ((x//16+y//16+frame)%2)*200
                        img[y,x] = [c, (c+hue)%256, 200-c%200]
                    else:              # Diagonal stripes
                        val = ((x+y+frame*5+seq*3)%64)*4
                        img[y,x] = [val, (val+hue)%256, 255-val]

            Image.fromarray(img).save(
                f"{folder}/seq{seq:04d}_f{frame:03d}.jpg"
            )
            count += 1

        if count % 500 == 0:
            print(f"    {count}/{target} video frames")

    print(f"  Video frames ready: {count}")


def generate_spectrograms(target=5000):
    """Generate diverse audio spectrogram images"""
    folder = "training_data/spectrograms"
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder)
                    if f.endswith('.jpg')])
    if existing >= target:
        print(f"  Spectrograms: {existing} already ready")
        return

    print(f"  Generating spectrograms (target: {target})...")
    t = np.linspace(0, 2, 256*256)

    for i in range(existing, target):
        pattern = np.random.randint(0, 6)

        if pattern == 0:   # Pure tones
            freqs  = np.random.uniform(0.5, 20, size=5)
            amps   = np.random.uniform(0.1, 1.0, size=5)
            signal = sum(a*np.sin(2*np.pi*f*t)
                        for f, a in zip(freqs, amps))
        elif pattern == 1: # Noise burst
            signal = np.random.randn(256*256)
            signal[:256*64] *= 3
        elif pattern == 2: # Chirp sweep
            signal = np.sin(2*np.pi*(1+10*t)*t)
        elif pattern == 3: # Harmonic series
            base   = np.random.uniform(1, 5)
            signal = sum(np.sin(2*np.pi*base*k*t)/k
                        for k in range(1, 8))
        elif pattern == 4: # Beat frequency
            f1     = np.random.uniform(5, 15)
            f2     = f1 + np.random.uniform(0.5, 2)
            signal = np.sin(2*np.pi*f1*t)+np.sin(2*np.pi*f2*t)
        else:              # AM modulation
            fc     = np.random.uniform(10, 20)
            fm     = np.random.uniform(0.5, 3)
            signal = (1+0.5*np.sin(2*np.pi*fm*t))*np.sin(2*np.pi*fc*t)

        signal += 0.05 * np.random.randn(len(signal))
        signal  = signal / (np.max(np.abs(signal)) + 1e-8)
        data    = signal.reshape(256, 256)

        r = ((data+1)/2*255).astype(np.uint8)
        g = (np.abs(data)*255).astype(np.uint8)
        b = ((1-np.abs(data))*255).astype(np.uint8)

        Image.fromarray(np.stack([r,g,b], axis=2)).save(
            f"{folder}/spec_{i:04d}.jpg"
        )
        if (i+1) % 500 == 0:
            print(f"    {i+1}/{target} spectrograms")

    print(f"  Spectrograms ready: {target}")


def generate_documents(target=5000):
    """Generate diverse synthetic document page images"""
    folder = "training_data/documents"
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder)
                    if f.endswith('.jpg')])
    if existing >= target:
        print(f"  Documents: {existing} already ready")
        return

    print(f"  Generating document images (target: {target})...")

    for i in range(existing, target):
        doc_type = np.random.randint(0, 6)
        bg       = np.random.randint(235, 256)
        img      = Image.new('RGB', (256,256), (bg,bg,bg))
        draw     = ImageDraw.Draw(img)

        if doc_type == 0:   # Text document
            draw.rectangle(
                [15,10,np.random.randint(150,240),24],
                fill=(20,20,20)
            )
            for line in range(np.random.randint(10,20)):
                y  = 35 + line*13
                lw = np.random.randint(30,230)
                gv = np.random.randint(20,80)
                draw.rectangle([15,y,lw,y+6], fill=(gv,gv,gv))

        elif doc_type == 1: # Invoice table
            for row in range(8):
                y = 20 + row*28
                draw.rectangle([10,y,246,y+20],
                               outline=(100,100,100), width=1)
                for col in [10,80,150,200]:
                    draw.rectangle([col,y,col+60,y+20],
                                  outline=(150,150,150), width=1)

        elif doc_type == 2: # Mixed content
            draw.rectangle([10,10,246,50], fill=(200,220,255))
            for line in range(6):
                y = 60+line*15
                draw.rectangle(
                    [10,y,np.random.randint(100,240),y+8],
                    fill=(50,50,50)
                )
            draw.rectangle([10,160,120,240], fill=(180,180,200))

        elif doc_type == 3: # Bar chart
            for bar in range(8):
                x = 20+bar*28
                h = np.random.randint(20,150)
                color = tuple(np.random.randint(50,200,3).tolist())
                draw.rectangle([x,256-h,x+20,246], fill=color)
            draw.rectangle([10,246,246,248], fill=(0,0,0))

        elif doc_type == 4: # Form fields
            for field in range(6):
                y = 20+field*36
                draw.rectangle([10,y,100,y+12],
                               fill=(80,80,80))
                draw.rectangle([110,y,246,y+20],
                               outline=(150,150,150), width=1)

        else:               # Newspaper layout
            draw.rectangle([10,5,246,40], fill=(30,30,30))
            draw.rectangle([10,45,120,160], fill=(200,200,200))
            for line in range(8):
                y = 45+line*14
                draw.rectangle(
                    [130,y,np.random.randint(200,240),y+8],
                    fill=(60,60,60)
                )

        img.save(f"{folder}/doc_{i:04d}.jpg")
        if (i+1) % 500 == 0:
            print(f"    {i+1}/{target} documents")

    print(f"  Documents ready: {target}")


# ── DATA LOADER ──────────────────────────────

def load_all_training_data(max_per_type=1000):
    """Load samples from all training folders"""
    folders = [
        ("training_images",            "Images"),
        ("training_data/video_frames", "Video "),
        ("training_data/spectrograms", "Audio "),
        ("training_data/documents",    "Docs  "),
    ]
    all_tensors = []
    for folder, name in folders:
        if not os.path.exists(folder):
            print(f"  {name}: folder missing")
            continue
        count = 0
        files = [f for f in os.listdir(folder)
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

        print(f"  {name}: {count:,} samples loaded")
    return all_tensors


# ── TRAINING LOOP ────────────────────────────

def train(epochs=2000, batch_size=32):
    print("\n" + "="*55)
    print("UNIVERSAL FOUNDATION TRAINER v5.2 FINAL")
    print("Images + Video + Audio + Documents")
    print(f"Device: {DEVICE}")
    print("Inventor: Rohit Kalu Sasane, Pune India 2026")
    print("="*55)

    # Generate all training data
    print("\nPreparing training data...")
    download_images_cifar(target=5000)
    generate_video_frames(target=5000)
    generate_spectrograms(target=5000)
    generate_documents(target=5000)

    # Load 1000 per type = 4000 total
    print("\nLoading training data...")
    tensors = load_all_training_data(max_per_type=1000)
    total   = len(tensors)
    print(f"\nTotal training samples: {total:,}")

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
            print("Continuing from existing weights")
        except:
            print("Starting fresh")
    else:
        print("Fresh training start")

    optimizer = torch.optim.Adam(
        foundation.parameters(), lr=0.001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    loss_fn   = nn.MSELoss()
    best_loss = float('inf')

    print(f"\nEpochs:    {epochs}")
    print(f"Batch:     {batch_size}")
    print(f"Samples:   {total:,}")
    print(f"GPU time:  ~2-3 hours\n")

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
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(
                foundation.parameters(), max_norm=1.0
            )
            optimizer.step()
            epoch_loss += loss.item()
            batches    += 1

        scheduler.step()
        avg = epoch_loss / max(batches, 1)

        if avg < best_loss:
            best_loss = avg
            atomic_save(foundation.state_dict(), SAVE_PATH)

        if (epoch+1) % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:>5}/{epochs} — "
                  f"Loss: {avg:.6f}  "
                  f"Best: {best_loss:.6f}  "
                  f"LR: {lr:.6f}")

    sz = os.path.getsize(SAVE_PATH)
    print(f"\n{'='*55}")
    print(f"Training complete!")
    print(f"Best loss:  {best_loss:.6f}")
    print(f"Saved:      {SAVE_PATH} ({sz/1024/1024:.1f}MB)")
    print(f"Samples:    {total:,} across all file types")
    print(f"Now run:    python foundation_v4.py")
    print(f"{'='*55}")


if __name__ == "__main__":
    if os.path.exists(SAVE_PATH):
        print(f"Weights exist: {SAVE_PATH}")
        print("Generating data only. Delete weights to retrain.\n")
        download_images_cifar(target=5000)
        generate_video_frames(target=5000)
        generate_spectrograms(target=5000)
        generate_documents(target=5000)
        print("\nAll data ready.")
    else:
        train(epochs=2000, batch_size=32)
