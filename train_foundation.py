import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import os
import sys
import tempfile
import random

# ============================================
# UNIVERSAL FOUNDATION TRAINER v5.7
# CURRICULUM LEARNING — ONE TYPE AT A TIME
# Phase 1: Images   — master visual patterns
# Phase 2: Video    — add motion knowledge
# Phase 3: Audio    — add frequency knowledge
# Phase 4: Documents — add layout knowledge
# NEW: Data augmentation — 3x effective dataset
# NEW: Validation split — prevents overfitting
# NEW: Early stopping — saves time on plateau
# NEW: Perceptual loss — better visual quality
# NEW: ReduceLROnPlateau — smarter learning rate
# Auto-resume from exact checkpoint
# Never loses progress on restart
# Inventor: Rohit Kalu Sasane, Pune India 2026
# ============================================

SAVE_PATH       = "foundation_v4_weights.pth"
CHECKPOINT_PATH = "foundation_v4_weights.pth.checkpoint"
IMAGE_SIZE      = 256
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHASES          = ["images", "video", "audio", "documents"]

# ── TRAINING CONFIG ───────────────────────────
# Change these to tune training
EPOCHS_PER_PHASE  = 2000   # max epochs per phase
BATCH_SIZE        = 64     # images per update
MAX_SAMPLES       = 500    # images loaded per phase
PATIENCE          = 150    # early stopping patience
VAL_SPLIT         = 0.15   # 15% validation set
AUGMENT_FACTOR    = 3      # how many augmented copies per image
PRINT_EVERY       = 50     # print progress every N epochs


# ── ARCHITECTURE ─────────────────────────────

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


# ── ATOMIC SAVE ──────────────────────────────

def atomic_save(state_dict, path):
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp  = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
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


# ── COMBINED LOSS — MSE + SSIM ───────────────

class CombinedLoss(nn.Module):
    """
    MSE + Structural Similarity loss.
    MSE:  pixel accuracy
    SSIM: structural/perceptual quality
    Combined gives better visual results than MSE alone.
    Weight: 0.8 MSE + 0.2 SSIM
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def ssim_loss(self, x, y):
        # Simplified SSIM — mean and variance based
        mu_x  = x.mean(dim=[2,3], keepdim=True)
        mu_y  = y.mean(dim=[2,3], keepdim=True)
        var_x = ((x - mu_x)**2).mean(dim=[2,3], keepdim=True)
        var_y = ((y - mu_y)**2).mean(dim=[2,3], keepdim=True)
        cov   = ((x - mu_x) * (y - mu_y)).mean(dim=[2,3], keepdim=True)
        c1, c2 = 0.01**2, 0.03**2
        ssim  = (
            (2*mu_x*mu_y + c1) * (2*cov + c2)
        ) / (
            (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        )
        return 1 - ssim.mean()

    def forward(self, pred, target):
        mse_loss  = self.mse(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        # 80% pixel accuracy + 20% structural quality
        return 0.8 * mse_loss + 0.2 * ssim_loss


# ── DATA AUGMENTATION ─────────────────────────

def augment_tensor(t):
    """
    Generate augmented versions of one image tensor.
    Returns list of augmented tensors.
    All augmentations preserve image content —
    just change appearance slightly.
    """
    img = Image.fromarray(
        (t.squeeze(0).permute(1,2,0).numpy() * 255).astype(np.uint8)
    )
    augmented = []

    # 1. Horizontal flip
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    augmented.append(
        torch.FloatTensor(
            np.array(flipped)/255.0
        ).permute(2,0,1).unsqueeze(0)
    )

    # 2. Random brightness
    factor = random.uniform(0.80, 1.20)
    bright = ImageEnhance.Brightness(img).enhance(factor)
    augmented.append(
        torch.FloatTensor(
            np.array(bright)/255.0
        ).permute(2,0,1).unsqueeze(0)
    )

    # 3. Random contrast
    factor = random.uniform(0.85, 1.15)
    contrast = ImageEnhance.Contrast(img).enhance(factor)
    augmented.append(
        torch.FloatTensor(
            np.array(contrast)/255.0
        ).permute(2,0,1).unsqueeze(0)
    )

    # 4. Slight rotation (±10 degrees)
    angle = random.uniform(-10, 10)
    rotated = img.rotate(angle, resample=Image.BILINEAR,
                         fillcolor=(128,128,128))
    augmented.append(
        torch.FloatTensor(
            np.array(rotated)/255.0
        ).permute(2,0,1).unsqueeze(0)
    )

    # 5. Random color shift
    factor = random.uniform(0.85, 1.15)
    colored = ImageEnhance.Color(img).enhance(factor)
    augmented.append(
        torch.FloatTensor(
            np.array(colored)/255.0
        ).permute(2,0,1).unsqueeze(0)
    )

    # 6. Vertical flip (less common but adds diversity)
    if random.random() > 0.5:
        vflipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        augmented.append(
            torch.FloatTensor(
                np.array(vflipped)/255.0
            ).permute(2,0,1).unsqueeze(0)
        )

    return augmented


# ── HIGH QUALITY IMAGES ───────────────────────

def download_images(target=1000):
    """
    STL10 — 96x96 real high quality photos
    10 categories: airplane car bird cat deer
                   dog horse monkey ship truck
    Upscaled to 256x256 with LANCZOS — excellent quality
    Download size: ~2.5GB but worth it
    """
    os.makedirs("training_images", exist_ok=True)
    existing = len([f for f in os.listdir("training_images")
                    if f.endswith('.jpg')])

    if existing >= target:
        print(f"  Images: {existing} already ready")
        return

    print(f"  Downloading high quality images (target: {target})...")
    print(f"  Using STL10 — 96x96 real photos")

    try:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.Resize((256, 256), Image.LANCZOS),
        ])

        count = existing

        ds_train = datasets.STL10(
            root='./stl10', split='train', download=True
        )
        ds_unlabeled = datasets.STL10(
            root='./stl10', split='unlabeled', download=True
        )

        print(f"  STL10 train:     {len(ds_train)} images")
        print(f"  STL10 unlabeled: {len(ds_unlabeled)} images")

        for ds_name, ds in [
            ("Train",     ds_train),
            ("Unlabeled", ds_unlabeled)
        ]:
            if count >= target:
                break
            for i in range(len(ds)):
                if count >= target:
                    break
                try:
                    img = ds[i][0]
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img)
                    img = img.convert('RGB')
                    img = img.resize((256,256), Image.LANCZOS)
                    img = ImageEnhance.Sharpness(img).enhance(1.3)
                    img = ImageEnhance.Contrast(img).enhance(1.1)
                    img = ImageEnhance.Color(img).enhance(1.1)
                    img.save(
                        f"training_images/stl_{count:05d}.jpg",
                        quality=97
                    )
                    count += 1
                    if count % 200 == 0:
                        print(f"    {count}/{target} images")
                except:
                    pass
            print(f"  {ds_name}: done")

        print(f"  Images ready: {count}")

    except Exception as e:
        print(f"  STL10 failed: {e}")
        print(f"  Trying CIFAR100 as backup...")
        try:
            import torchvision.datasets as datasets
            count = existing
            ds = datasets.CIFAR100(
                root='./cifar100', download=True, train=True
            )
            for i in range(len(ds)):
                if count >= target:
                    break
                img, _ = ds[i]
                img = img.resize((256,256), Image.LANCZOS)
                img = ImageEnhance.Sharpness(img).enhance(1.5)
                img.save(
                    f"training_images/img_{count:05d}.jpg",
                    quality=95
                )
                count += 1
                if count % 200 == 0:
                    print(f"    {count}/{target} images")
            print(f"  Images ready: {count}")
        except Exception as e2:
            print(f"  Backup failed: {e2}")


# ── HIGH QUALITY VIDEO FRAMES ─────────────────

def generate_video_frames(target=1000):
    """
    50 cinematic motion pattern types
    High quality with smooth interpolation
    """
    folder = "training_data/video_frames"
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder)
                    if f.endswith('.jpg')])
    if existing >= target:
        print(f"  Video frames: {existing} already ready")
        return

    print(f"  Generating {target} cinematic video frames...")

    patterns = [
        'sunset_gradient', 'ocean_wave', 'forest_light',
        'city_night', 'aurora', 'fire_glow',
        'water_ripple', 'starfield', 'nebula',
        'lava_flow', 'ice_crystal', 'sand_dune',
        'rain_drop', 'snow_fall', 'cloud_move',
        'lens_flare', 'bokeh', 'depth_field',
        'motion_blur', 'zoom_burst',
        'color_grade_warm', 'color_grade_cool',
        'color_grade_teal', 'color_grade_vintage',
        'film_grain', 'vignette', 'chromatic',
        'double_exposure', 'light_leak', 'prism',
        'kaleidoscope', 'mandala', 'fractal',
        'topology', 'wireframe', 'hologram',
        'scan_line', 'glitch', 'pixel_sort',
        'wave_interference', 'standing_wave',
        'spiral_galaxy', 'black_hole', 'corona',
        'bioluminescence', 'crystal_growth',
        'oil_slick', 'soap_bubble', 'prism_split',
        'thermal_cam', 'sonar_ping'
    ]

    count = existing
    frames_per_pattern = max(1, target // len(patterns))
    x_grid, y_grid = np.meshgrid(
        np.linspace(0, 1, 256),
        np.linspace(0, 1, 256)
    )

    for pat_idx, pattern in enumerate(patterns):
        if count >= target:
            break
        for frame in range(frames_per_pattern):
            if count >= target:
                break

            t  = frame / max(frames_per_pattern, 1)
            pi = np.pi
            x, y = x_grid, y_grid
            img = np.zeros((256, 256, 3), dtype=np.float32)

            if pattern == 'sunset_gradient':
                img[:,:,0] = 0.9 - y*0.3 + t*0.1
                img[:,:,1] = 0.4 + y*0.2 - t*0.1
                img[:,:,2] = 0.1 + (1-y)*0.4 + t*0.2

            elif pattern == 'ocean_wave':
                wave = np.sin(x*8*pi + t*2*pi) * 0.1
                depth = y + wave
                img[:,:,0] = np.clip(0.1 - depth*0.1, 0, 1)
                img[:,:,1] = np.clip(0.3 + depth*0.3, 0, 1)
                img[:,:,2] = np.clip(0.6 + depth*0.4, 0, 1)

            elif pattern == 'aurora':
                r = np.sin(x*3*pi + t*pi) * np.exp(-y*3)
                g = np.sin(x*5*pi + t*2*pi) * np.exp(-y*2)
                b = np.cos(x*4*pi + t*pi) * np.exp(-y*2.5)
                img[:,:,0] = np.clip(r*0.5+0.1, 0, 1)
                img[:,:,1] = np.clip(g*0.8+0.2, 0, 1)
                img[:,:,2] = np.clip(b*0.7+0.3, 0, 1)

            elif pattern == 'nebula':
                r = np.sin(x*pi*3 + y*pi*2 + t*pi)
                g = np.cos(x*pi*4 + t*pi*2)
                b = np.sin(y*pi*5 + t*pi*3)
                noise = np.random.rand(256,256) * 0.05
                img[:,:,0] = np.clip(r*0.5+0.4+noise, 0, 1)
                img[:,:,1] = np.clip(g*0.3+0.2+noise, 0, 1)
                img[:,:,2] = np.clip(b*0.6+0.5+noise, 0, 1)

            elif pattern == 'fire_glow':
                heat = np.sin(x*pi*4 + t*3*pi) * np.exp(-(1-y)*3)
                img[:,:,0] = np.clip(heat*0.8+0.6, 0, 1)
                img[:,:,1] = np.clip(heat*0.4+0.1, 0, 1)
                img[:,:,2] = np.clip(heat*0.1, 0, 1)

            elif pattern == 'starfield':
                np.random.seed(pat_idx*100 + frame)
                stars = np.random.rand(256,256)
                bright = (stars > 0.98).astype(np.float32)
                twinkle = bright * (0.8 + 0.2*np.sin(t*10*pi))
                img[:,:,0] = twinkle
                img[:,:,1] = twinkle
                img[:,:,2] = np.clip(twinkle + bright*0.2, 0, 1)

            elif pattern == 'kaleidoscope':
                cx = x - 0.5
                cy = y - 0.5
                r  = np.sqrt(cx**2 + cy**2)
                a  = np.arctan2(cy, cx) + t*pi
                a  = np.abs(((a % (pi/3)) - pi/6))
                img[:,:,0] = np.clip(np.sin(r*8*pi + a*6)*0.5+0.5, 0, 1)
                img[:,:,1] = np.clip(np.cos(r*6*pi + t*2*pi)*0.5+0.5, 0, 1)
                img[:,:,2] = np.clip(np.sin(a*8 + t*pi)*0.5+0.5, 0, 1)

            elif pattern == 'oil_slick':
                r = np.sqrt((x-0.5)**2 + (y-0.5)**2)
                h = (r*8 + t*2) % 1.0
                s = np.ones_like(h) * 0.9
                v = np.ones_like(h) * 0.8
                hi = (h*6).astype(int) % 6
                f  = h*6 - hi.astype(float)
                p  = v*(1-s)
                q  = v*(1-s*f)
                tk = v*(1-s*(1-f))
                for c_idx, (rc,gc,bc) in enumerate([
                    (v,tk,p),(q,v,p),(p,v,tk),
                    (p,q,v),(tk,p,v),(v,p,q)
                ]):
                    mask = (hi == c_idx)
                    img[:,:,0] += rc * mask
                    img[:,:,1] += gc * mask
                    img[:,:,2] += bc * mask

            elif pattern == 'glitch':
                base = x.copy()
                for line in range(0, 256, np.random.randint(8,32)):
                    shift = int(np.random.randint(-20,20) * t)
                    if 0 <= line < 256:
                        img[line,:,0] = np.roll(base[line], shift)
                        img[line,:,1] = np.roll(base[line], shift//2)
                        img[line,:,2] = np.roll(base[line], -shift//2)

            elif pattern == 'thermal_cam':
                heat = np.sin(x*pi*3)*np.cos(y*pi*2) + t*0.5
                heat = (heat - heat.min()) / (heat.max()-heat.min()+1e-8)
                img[:,:,0] = np.clip(heat*2, 0, 1)
                img[:,:,1] = np.clip(1-np.abs(heat-0.5)*2, 0, 1)
                img[:,:,2] = np.clip((1-heat)*2, 0, 1)

            else:
                r = np.sin(x*pi*(pat_idx%5+2) + t*2*pi)*0.5+0.5
                g = np.cos(y*pi*(pat_idx%4+3) + t*pi)*0.5+0.5
                b = np.sin((x+y)*pi*(pat_idx%3+2) + t*3*pi)*0.5+0.5
                img[:,:,0] = r
                img[:,:,1] = g
                img[:,:,2] = b

            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            pil_img = pil_img.filter(ImageFilter.SMOOTH)
            pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.2)
            pil_img.save(
                f"{folder}/pat{pat_idx:02d}_f{frame:03d}.jpg",
                quality=95
            )
            count += 1

    print(f"  Video frames ready: {count}")


# ── HIGH QUALITY AUDIO SPECTROGRAMS ──────────

def generate_spectrograms(target=1000):
    """
    50 diverse rich audio patterns
    RGB spectrograms with high color depth
    """
    folder = "training_data/spectrograms"
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder)
                    if f.endswith('.jpg')])
    if existing >= target:
        print(f"  Spectrograms: {existing} already ready")
        return

    print(f"  Generating {target} rich audio spectrograms...")

    audio_types = [
        'pure_tone','dual_tone','triple_tone','chord','arpeggio',
        'octave','fifth','fourth','major_scale','minor_scale',
        'white_noise','pink_noise','brown_noise','burst_noise','impulse',
        'bass_heavy','treble_heavy','mid_range','full_spectrum','sub_bass',
        'chirp_up','chirp_down','chirp_fast','chirp_slow','chirp_complex',
        'am_slow','am_fast','fm_slow','fm_fast','pm_mod',
        'ring_mod','tremolo','vibrato','chorus','flanger',
        'rain','thunder','wind','ocean','crowd',
        'harmonic_rich','harmonic_sparse','inharmonic','beating','resonance',
        'speech_vowel','speech_consonant','whisper','shout','music_mix',
        'echo_short','echo_long','reverb_room','reverb_hall','reverb_plate'
    ]

    t_arr = np.linspace(0, 4, 256*256)
    count = existing

    for type_idx, audio_type in enumerate(audio_types):
        if count >= target:
            break
        samples_per_type = max(1, target // len(audio_types))

        for variant in range(samples_per_type):
            if count >= target:
                break

            fb = np.random.uniform(2, 30)

            if audio_type == 'pure_tone':
                s = np.sin(2*np.pi*fb*t_arr)
            elif audio_type == 'dual_tone':
                s = (0.6*np.sin(2*np.pi*fb*t_arr) +
                     0.4*np.sin(2*np.pi*fb*1.5*t_arr))
            elif audio_type == 'chord':
                ratios = [1, 5/4, 3/2, 2, 5/2]
                s = sum(np.sin(2*np.pi*fb*r*t_arr)/5
                       for r in ratios)
            elif audio_type == 'arpeggio':
                scale  = [1, 5/4, 3/2, 2]
                chunks = np.array_split(t_arr, 4)
                s = np.concatenate([
                    np.sin(2*np.pi*fb*r*c)
                    for r, c in zip(scale, chunks)
                ])
            elif audio_type == 'major_scale':
                notes  = [1,9/8,5/4,4/3,3/2,5/3,15/8,2]
                chunks = np.array_split(t_arr, 8)
                s = np.concatenate([
                    np.sin(2*np.pi*fb*n*c)
                    for n, c in zip(notes, chunks)
                ])
            elif audio_type == 'white_noise':
                s = np.random.randn(len(t_arr))
            elif audio_type == 'pink_noise':
                w = np.random.randn(len(t_arr))
                s = np.cumsum(w) / (np.arange(1,len(t_arr)+1)**0.5)
            elif audio_type == 'brown_noise':
                w = np.random.randn(len(t_arr))
                s = np.cumsum(w)
            elif audio_type == 'chirp_up':
                s = np.sin(2*np.pi*(fb + 20*t_arr)*t_arr)
            elif audio_type == 'chirp_down':
                s = np.sin(2*np.pi*(fb*5 - 20*t_arr)*t_arr)
            elif audio_type == 'am_slow':
                s = np.sin(2*np.pi*fb*10*t_arr) * (1+0.8*np.sin(2*np.pi*0.5*t_arr))
            elif audio_type == 'fm_fast':
                s = np.sin(2*np.pi*fb*t_arr + 8*np.sin(2*np.pi*5*t_arr))
            elif audio_type == 'harmonic_rich':
                s = sum(np.sin(2*np.pi*fb*k*t_arr)/k for k in range(1,16))
            elif audio_type == 'beating':
                s = np.sin(2*np.pi*fb*t_arr)+np.sin(2*np.pi*(fb+0.3)*t_arr)
            elif audio_type == 'speech_vowel':
                f1,f2,f3 = fb,fb*3,fb*6
                s = (np.sin(2*np.pi*f1*t_arr)*0.5 +
                     np.sin(2*np.pi*f2*t_arr)*0.3 +
                     np.sin(2*np.pi*f3*t_arr)*0.2)
                s *= (1+0.4*np.sin(2*np.pi*4*t_arr))
            elif audio_type == 'echo_short':
                base  = np.sin(2*np.pi*fb*t_arr)
                delay = np.roll(base, 1000)
                s     = base + 0.5*delay + 0.25*np.roll(delay,1000)
            elif audio_type == 'reverb_hall':
                base = np.sin(2*np.pi*fb*t_arr)
                s    = sum(0.8**i * np.roll(base, i*500)
                          for i in range(8))
            else:
                freqs = np.random.uniform(1, 30, size=6)
                amps  = np.random.uniform(0.1, 1.0, size=6)
                s = sum(a*np.sin(2*np.pi*f*t_arr)
                       for f,a in zip(freqs,amps))

            s += 0.01*np.random.randn(len(s))
            s  = s / (np.max(np.abs(s)) + 1e-8)
            d  = s.reshape(256, 256)

            r_ch = ((d+1)/2*255).astype(np.uint8)
            g_ch = (np.abs(d)*255).astype(np.uint8)
            b_ch = np.clip(
                (1-np.abs(d))*180 + np.abs(np.roll(d,8,1))*75,
                0, 255
            ).astype(np.uint8)

            img = Image.fromarray(np.stack([r_ch,g_ch,b_ch], axis=2))
            img = img.filter(ImageFilter.SMOOTH_MORE)
            img = ImageEnhance.Contrast(img).enhance(1.2)
            img.save(
                f"{folder}/type{type_idx:02d}_v{variant:03d}.jpg",
                quality=95
            )
            count += 1

    print(f"  Spectrograms ready: {count}")


# ── HIGH QUALITY DOCUMENT LAYOUTS ────────────

def generate_documents(target=1000):
    """
    50 diverse professional document layouts
    High quality with realistic proportions
    """
    folder = "training_data/documents"
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder)
                    if f.endswith('.jpg')])
    if existing >= target:
        print(f"  Documents: {existing} already ready")
        return

    print(f"  Generating {target} professional document layouts...")

    doc_types = [
        'report','letter','memo','essay','article',
        'resume','contract','manual','thesis','notes',
        'spreadsheet','invoice','receipt','statement','budget',
        'table_data','comparison','schedule','calendar','timeline',
        'presentation','poster','flyer','brochure','certificate',
        'infographic','diagram','flowchart','org_chart','mind_map',
        'research_paper','lab_report','equation_sheet','graph_sheet','data_table',
        'business_card','letterhead','form','questionnaire','feedback_form',
        'newspaper','magazine','book_page','legal_doc','medical_record',
        'blueprint','map_legend','music_sheet','code_listing','comic_layout'
    ]

    count = existing
    samples_per_type = max(1, target // len(doc_types))

    for type_idx, doc_type in enumerate(doc_types):
        if count >= target:
            break

        for variant in range(samples_per_type):
            if count >= target:
                break

            bg     = np.random.randint(245, 256)
            img    = Image.new('RGB', (256,256), (bg,bg,bg))
            draw   = ImageDraw.Draw(img)
            accent = tuple(np.random.randint(20,180,3).tolist())
            dark   = tuple(np.random.randint(20,60,3).tolist())
            light  = tuple(np.random.randint(200,240,3).tolist())

            if doc_type in ['report','essay','article','thesis','manual']:
                draw.rectangle([0,0,256,38], fill=accent)
                draw.rectangle([10,8,220,30], fill=(255,255,255))
                draw.rectangle([10,8,180,20], fill=light)
                for line in range(14):
                    y  = 48 + line*13
                    lw = np.random.randint(160,248)
                    gv = np.random.randint(50,90)
                    h  = np.random.randint(5,9)
                    draw.rectangle([10,y,lw,y+h], fill=(gv,gv,gv))
                draw.rectangle([100,248,156,255], fill=(180,180,180))

            elif doc_type in ['resume','letter','contract']:
                draw.rectangle([0,0,256,55], fill=accent)
                draw.rectangle([10,10,160,45], fill=(255,255,255))
                draw.rectangle([0,55,6,256], fill=accent)
                for s_idx in range(3):
                    y = 65 + s_idx*62
                    draw.rectangle([12,y,110,y+11], fill=dark)
                    draw.rectangle([12,y+15,246,y+16],
                                   fill=(200,200,200))
                    for line in range(3):
                        lw = np.random.randint(120,240)
                        draw.rectangle(
                            [15, y+20+line*13,
                             lw, y+28+line*13],
                            fill=(120,120,120)
                        )

            elif doc_type in ['invoice','receipt','statement','budget']:
                draw.rectangle([0,0,256,45], fill=accent)
                draw.rectangle([10,8,180,37], fill=(255,255,255))
                draw.rectangle([5,50,251,65], fill=dark)
                for row in range(7):
                    y = 70 + row*24
                    fill_color = light if row%2==0 else (bg,bg,bg)
                    draw.rectangle([5,y,251,y+22], fill=fill_color)
                    for col in [5,75,140,195]:
                        w = np.random.randint(25,55)
                        draw.rectangle([col+3,y+5,col+w,y+16],
                                       fill=(100,100,100))
                draw.rectangle([5,240,251,255], fill=accent)

            elif doc_type in ['spreadsheet','table_data','comparison']:
                draw.rectangle([0,0,256,20], fill=accent)
                for row in range(11):
                    y = 20 + row*22
                    c = light if row%2==0 else (bg,bg,bg)
                    draw.rectangle([0,y,256,y+22], fill=c)
                    draw.rectangle([0,y,256,y+1], fill=(180,180,180))
                for col in range(5):
                    x = col*52
                    draw.rectangle([x,0,x+1,256], fill=(180,180,180))
                    for row in range(1,11):
                        y = 20+row*22
                        w = np.random.randint(15,45)
                        draw.rectangle([x+3,y+5,x+w,y+16],
                                       fill=(80,80,80))

            elif doc_type in ['graph_sheet','infographic','data_table']:
                draw.rectangle([0,0,256,256], fill=(248,250,255))
                draw.line([(25,225),(240,225)], fill=(0,0,0), width=2)
                draw.line([(25,15),(25,225)], fill=(0,0,0), width=2)
                colors = [
                    (220,60,60),(60,160,220),(60,200,100),
                    (220,160,60),(160,60,220)
                ]
                for i in range(8):
                    x = 40 + i*25
                    h = np.random.randint(30,190)
                    draw.rectangle([x,225-h,x+18,225],
                                   fill=colors[i%5])
                for i in range(5):
                    y = 225 - i*40
                    draw.line([(22,y),(240,y)],
                              fill=(200,200,200), width=1)

            elif doc_type in ['flowchart','diagram','org_chart','mind_map']:
                draw.rectangle([88,8,168,36], fill=accent)
                draw.rectangle([90,10,166,34], fill=(255,255,255))
                draw.line([(128,36),(128,55)], fill=dark, width=2)
                positions = [(20,55),(88,55),(156,55)]
                for px in positions:
                    draw.rectangle(
                        [px[0],px[1],px[0]+60,px[1]+28],
                        fill=light, outline=dark, width=1
                    )
                    draw.line(
                        [(px[0]+30,px[1]+28),(px[0]+30,px[1]+48)],
                        fill=dark, width=1
                    )
                    draw.rectangle(
                        [px[0],px[1]+48,px[0]+60,px[1]+76],
                        fill=(220,240,200), outline=dark, width=1
                    )

            elif doc_type in ['newspaper','magazine']:
                draw.rectangle([0,0,256,32], fill=(15,15,15))
                draw.rectangle([8,6,170,26], fill=(255,255,255))
                draw.rectangle([0,32,256,33], fill=(180,0,0))
                draw.rectangle([128,38,130,248], fill=(180,180,180))
                for col in range(2):
                    xo = col*133
                    draw.rectangle([xo+5,38,xo+122,95],
                                   fill=(210,210,210))
                    for line in range(9):
                        y  = 100 + line*14
                        w  = np.random.randint(70,118)
                        gv = np.random.randint(40,90)
                        draw.rectangle([xo+5,y,xo+w,y+9],
                                       fill=(gv,gv,gv))

            elif doc_type in ['presentation','poster','flyer']:
                bg_c = tuple(np.random.randint(80,180,3).tolist())
                img  = Image.new('RGB', (256,256), bg_c)
                draw = ImageDraw.Draw(img)
                draw.rectangle([0,0,256,60], fill=(0,0,0))
                draw.rectangle([8,8,248,52], fill=(255,255,255))
                draw.rectangle([8,68,248,68+np.random.randint(40,80)],
                               fill=(255,255,255))
                for block in range(3):
                    x = 8 + block*83
                    draw.rectangle([x,165,x+75,235],
                                   fill=(255,255,255))
                draw.rectangle([0,242,256,256], fill=(30,30,30))

            elif doc_type in ['blueprint','map_legend']:
                img  = Image.new('RGB', (256,256), (30,50,100))
                draw = ImageDraw.Draw(img)
                for i in range(0,256,16):
                    draw.line([(0,i),(256,i)], fill=(50,80,140), width=1)
                    draw.line([(i,0),(i,256)], fill=(50,80,140), width=1)
                shapes = [
                    (20,20,100,80), (120,20,220,80),
                    (20,100,80,180), (100,100,200,200),
                    (210,100,250,180)
                ]
                for rect in shapes:
                    draw.rectangle(rect, outline=(100,200,255), width=2)
                for i in range(5):
                    y = 20+i*45
                    draw.line([(20,y),(220,y)],
                              fill=(255,255,100), width=1)

            elif doc_type in ['code_listing']:
                img  = Image.new('RGB', (256,256), (30,30,40))
                draw = ImageDraw.Draw(img)
                line_colors = [
                    (100,200,100),(150,150,255),(255,180,50),
                    (200,200,200),(255,100,100),(100,255,200)
                ]
                for line in range(16):
                    y      = 10 + line*15
                    indent = np.random.randint(0,4) * 8
                    lw     = np.random.randint(60,220)
                    col    = line_colors[line%len(line_colors)]
                    draw.rectangle(
                        [10+indent, y, 10+indent+lw, y+9],
                        fill=col
                    )

            elif doc_type in ['music_sheet']:
                draw.rectangle([0,0,256,256], fill=(255,252,240))
                for staff in range(3):
                    y_base = 30 + staff*80
                    for line in range(5):
                        y = y_base + line*8
                        draw.line([(10,y),(246,y)],
                                  fill=(0,0,0), width=1)
                    for note in range(np.random.randint(4,8)):
                        x  = 20 + note*30
                        yn = y_base + np.random.randint(0,40)
                        draw.ellipse([x,yn,x+8,yn+6],
                                     fill=(0,0,0))

            else:
                draw.rectangle([0,0,256,32], fill=accent)
                for section in range(4):
                    y = 40 + section*52
                    draw.rectangle([8,y,248,y+48],
                                   outline=(180,180,180), width=1)
                    draw.rectangle([12,y+5,210,y+16],
                                   fill=dark)
                    for line in range(2):
                        lw = np.random.randint(80,235)
                        draw.rectangle(
                            [12, y+22+line*13,
                             lw, y+30+line*13],
                            fill=(140,140,140)
                        )

            img = img.filter(ImageFilter.SMOOTH)
            img = ImageEnhance.Sharpness(img).enhance(1.4)
            img = ImageEnhance.Contrast(img).enhance(1.1)
            img.save(
                f"{folder}/type{type_idx:02d}_v{variant:03d}.jpg",
                quality=97
            )
            count += 1

    print(f"  Documents ready: {count}")


# ── DATA LOADER WITH AUGMENTATION ────────────

def load_phase_data(phase, max_samples=500):
    """
    Load training data for one phase.
    Applies augmentation to multiply effective dataset.
    Splits into train/val sets.
    Returns: train_tensors, val_tensors
    """
    folder_map = {
        "images":    "training_images",
        "video":     "training_data/video_frames",
        "audio":     "training_data/spectrograms",
        "documents": "training_data/documents",
    }
    folder = folder_map[phase]
    if not os.path.exists(folder):
        print(f"  Folder missing: {folder}")
        return [], []

    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(files)
    files = files[:max_samples]

    # Load base tensors
    base_tensors = []
    for fname in files:
        try:
            img = Image.open(
                os.path.join(folder, fname)
            ).convert('RGB').resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
            )
            t = torch.FloatTensor(
                np.array(img)/255.0
            ).permute(2,0,1).unsqueeze(0)
            base_tensors.append(t)
        except:
            pass

    if not base_tensors:
        return [], []

    # ── Validation split — before augmentation ──
    # Val set uses ONLY original images — no augmentation
    # This gives honest evaluation on unseen variations
    n_val   = max(1, int(len(base_tensors) * VAL_SPLIT))
    n_train = len(base_tensors) - n_val
    random.shuffle(base_tensors)
    val_tensors   = base_tensors[:n_val]
    train_base    = base_tensors[n_val:]

    # ── Augmentation on train set only ──
    train_tensors = list(train_base)
    for t in train_base:
        augmented = augment_tensor(t)
        train_tensors.extend(augmented)

    random.shuffle(train_tensors)

    print(f"  {phase.capitalize()}:")
    print(f"    Base:  {len(base_tensors)} images")
    print(f"    Train: {len(train_tensors)} "
          f"(after augmentation)")
    print(f"    Val:   {len(val_tensors)} "
          f"(original only — honest eval)")

    return train_tensors, val_tensors


# ── PREPARE DATA FOR PHASE ────────────────────

def prepare_phase_data(phase):
    if phase == "images":
        download_images(target=1000)
    elif phase == "video":
        generate_video_frames(target=1000)
    elif phase == "audio":
        generate_spectrograms(target=1000)
    elif phase == "documents":
        generate_documents(target=1000)


# ── VALIDATION LOSS ───────────────────────────

def compute_val_loss(foundation, val_tensors,
                     loss_fn, batch_size=64):
    """
    Compute loss on validation set.
    No gradient computation — faster.
    """
    foundation.eval()
    total_loss = 0.0
    batches    = 0
    with torch.no_grad():
        for i in range(0, len(val_tensors), batch_size):
            batch_list = val_tensors[i:i+batch_size]
            if not batch_list:
                continue
            batch = torch.cat(batch_list).to(DEVICE)
            out, *_ = foundation(batch)
            loss = loss_fn(out, batch)
            total_loss += loss.item()
            batches    += 1
    return total_loss / max(batches, 1)


# ── SINGLE PHASE TRAINING ─────────────────────

def train_phase(phase, epochs=EPOCHS_PER_PHASE,
                batch_size=BATCH_SIZE):
    """
    Train foundation on one file type.
    Features:
      - Data augmentation (3x dataset)
      - Validation split (honest evaluation)
      - Early stopping (prevents overfitting)
      - Combined MSE+SSIM loss (better quality)
      - ReduceLROnPlateau (smarter LR)
      - Checkpoint resume (never loses progress)
    """
    ckpt_path = f"{CHECKPOINT_PATH}.{phase}"

    print(f"\n{'='*55}")
    print(f"PHASE: {phase.upper()}")
    print(f"Max epochs: {epochs}  "
          f"Batch: {batch_size}  "
          f"Patience: {PATIENCE}")
    print(f"Device: {DEVICE}")
    print(f"{'='*55}")

    print(f"\nPreparing {phase} data...")
    prepare_phase_data(phase)
    train_tensors, val_tensors = load_phase_data(
        phase, max_samples=MAX_SAMPLES
    )

    if not train_tensors:
        print(f"  No data for phase {phase}. Skipping.")
        return

    # ── Build model ───────────────────────────
    foundation = Foundation().to(DEVICE)

    # ── Combined loss — better visual quality ──
    loss_fn = CombinedLoss().to(DEVICE)

    # ── Adam optimizer ────────────────────────
    optimizer = torch.optim.Adam(
        foundation.parameters(),
        lr=0.001, weight_decay=1e-5
    )

    # ── ReduceLROnPlateau ─────────────────────
    # Drops LR when val_loss stops improving
    # More responsive than CosineAnnealing
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # minimize val_loss
        factor=0.5,        # halve LR on plateau
        patience=30,       # wait 30 epochs before dropping
        min_lr=1e-6,       # never go below this
        verbose=False
    )

    best_train_loss = float('inf')
    best_val_loss   = float('inf')
    start_epoch     = 0
    no_improve      = 0   # early stopping counter

    # ── Load existing weights ─────────────────
    if os.path.exists(SAVE_PATH):
        try:
            foundation.load_state_dict(
                torch.load(SAVE_PATH, map_location=DEVICE,
                           weights_only=True)
            )
            print(f"  Weights loaded")
        except:
            print(f"  Weights incompatible — starting fresh")
    else:
        print(f"  Fresh start")

    # ── Load phase checkpoint ─────────────────
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(
                ckpt_path, map_location=DEVICE,
                weights_only=False
            )
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch     = ckpt['epoch'] + 1
            best_val_loss   = ckpt['best_val_loss']
            best_train_loss = ckpt.get('best_train_loss',
                                       float('inf'))
            no_improve      = ckpt.get('no_improve', 0)
            print(f"  Resuming from epoch {start_epoch}/{epochs}"
                  f" — best val: {best_val_loss:.6f}"
                  f" — no_improve: {no_improve}/{PATIENCE}")
        except Exception as e:
            print(f"  No checkpoint — fresh optimizer ({e})")
    else:
        print(f"  No checkpoint — fresh optimizer")

    if start_epoch >= epochs:
        print(f"  Already completed {epochs} epochs.")
        print(f"  Increase EPOCHS_PER_PHASE to train more.")
        return

    print(f"\n  Epochs:  {start_epoch} → {epochs}")
    print(f"  Train:   {len(train_tensors):,} samples")
    print(f"  Val:     {len(val_tensors):,} samples")
    print()

    for epoch in range(start_epoch, epochs):

        # ── TRAINING PASS ─────────────────────
        foundation.train()
        epoch_loss = 0.0
        random.shuffle(train_tensors)
        batches = 0

        for i in range(0, len(train_tensors), batch_size):
            batch_list = train_tensors[i:i+batch_size]
            if not batch_list:
                continue
            batch = torch.cat(batch_list).to(DEVICE)
            optimizer.zero_grad()
            out, g, r, d = foundation(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(
                foundation.parameters(), max_norm=1.0
            )
            optimizer.step()
            epoch_loss += loss.item()
            batches    += 1

        train_avg = epoch_loss / max(batches, 1)

        # ── VALIDATION PASS ───────────────────
        val_avg = compute_val_loss(
            foundation, val_tensors, loss_fn, batch_size
        )
        foundation.train()

        # ── LR SCHEDULER step ─────────────────
        # Uses val_loss for smarter LR adjustment
        scheduler.step(val_avg)

        # ── SAVE if val_loss improved ──────────
        if val_avg < best_val_loss:
            best_val_loss   = val_avg
            best_train_loss = train_avg
            no_improve      = 0
            atomic_save(foundation.state_dict(), SAVE_PATH)
            torch.save({
                'epoch':           epoch,
                'best_val_loss':   best_val_loss,
                'best_train_loss': best_train_loss,
                'no_improve':      no_improve,
                'optimizer':       optimizer.state_dict(),
                'scheduler':       scheduler.state_dict(),
                'phase':           phase,
            }, ckpt_path)
        else:
            no_improve += 1

        # ── PRINT PROGRESS ────────────────────
        if (epoch+1) % PRINT_EVERY == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:>5}/{epochs} — "
                  f"Train: {train_avg:.6f}  "
                  f"Val: {val_avg:.6f}  "
                  f"Best: {best_val_loss:.6f}  "
                  f"LR: {lr:.7f}  "
                  f"Patience: {no_improve}/{PATIENCE}")

        # ── EARLY STOPPING ────────────────────
        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch+1}")
            print(f"  Val loss not improved for "
                  f"{PATIENCE} epochs")
            print(f"  Best val loss: {best_val_loss:.6f}")
            break

    sz = os.path.getsize(SAVE_PATH)
    print(f"\n  Phase {phase} complete!")
    print(f"  Best val loss:   {best_val_loss:.6f}")
    print(f"  Best train loss: {best_train_loss:.6f}")
    print(f"  Weights:         {sz/1024/1024:.1f}MB")
    print(f"  Augmented data:  {len(train_tensors):,} samples")


# ── CURRICULUM TRAINER ────────────────────────

def train_curriculum(epochs_per_phase=EPOCHS_PER_PHASE,
                     batch_size=BATCH_SIZE):
    print("\n" + "="*55)
    print("UNIVERSAL FOUNDATION TRAINER v5.7")
    print("CURRICULUM LEARNING — one type at a time")
    print(f"Phases:        {' → '.join(PHASES)}")
    print(f"Max epochs:    {epochs_per_phase} per phase")
    print(f"Early stop:    {PATIENCE} patience")
    print(f"Augmentation:  {AUGMENT_FACTOR}x dataset")
    print(f"Val split:     {int(VAL_SPLIT*100)}%")
    print(f"Loss:          MSE + SSIM combined")
    print(f"LR schedule:   ReduceLROnPlateau")
    print(f"Device:        {DEVICE}")
    print("Inventor: Rohit Kalu Sasane, Pune India 2026")
    print("="*55)

    for phase in PHASES:
        train_phase(phase,
                    epochs=epochs_per_phase,
                    batch_size=batch_size)

    print("\n" + "="*55)
    print("ALL PHASES COMPLETE")
    print(f"Weights: {SAVE_PATH}")
    print("="*55)


# ── MAIN ─────────────────────────────────────

if __name__ == "__main__":
    # Usage:
    #   python train_foundation.py            — all phases
    #   python train_foundation.py images     — images only
    #   python train_foundation.py video      — video only
    #   python train_foundation.py audio      — audio only
    #   python train_foundation.py documents  — documents only

    if len(sys.argv) > 1:
        phase = sys.argv[1].lower()
        if phase not in PHASES:
            print(f"Unknown phase: {phase}")
            print(f"Valid: {', '.join(PHASES)}")
        else:
            train_phase(phase)
    else:
        train_curriculum()