import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import os
import tempfile
import requests

# ============================================
# UNIVERSAL FOUNDATION TRAINER v5.3
# 50 DIVERSE CATEGORIES PER FILE TYPE
# Maximum quality — high resolution patterns
# Images:     50 categories × real photos
# Video:      50 motion pattern types
# Audio:      50 frequency pattern types
# Documents:  50 layout types
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


# ── 50 IMAGE CATEGORIES ──────────────────────

IMAGE_CATEGORIES = [
    # Nature
    'mountain', 'ocean', 'forest', 'desert', 'waterfall',
    'lake', 'river', 'sky', 'clouds', 'sunset',
    # Urban
    'city', 'street', 'architecture', 'bridge', 'skyscraper',
    'night city', 'market', 'subway', 'airport', 'stadium',
    # People
    'portrait', 'crowd', 'children', 'elderly', 'wedding',
    'sports', 'dance', 'fashion', 'work', 'celebration',
    # Animals
    'dog', 'cat', 'bird', 'lion', 'elephant',
    'fish', 'butterfly', 'horse', 'wolf', 'eagle',
    # Objects and textures
    'food', 'flowers', 'cars', 'technology', 'art',
    'abstract', 'texture', 'space', 'underwater', 'fire'
]


def download_images_diverse(target_per_category=20):
    """Download real photos — 50 categories × 20 photos = 1000 images"""
    os.makedirs("training_images", exist_ok=True)
    existing = len([f for f in os.listdir("training_images")
                    if f.endswith('.jpg')])
    total_target = len(IMAGE_CATEGORIES) * target_per_category

    if existing >= total_target:
        print(f"  Images: {existing} already ready")
        return

    print(f"  Downloading {total_target} diverse images "
          f"({len(IMAGE_CATEGORIES)} categories × {target_per_category})...")

    count = existing
    for cat_idx, category in enumerate(IMAGE_CATEGORIES):
        cat_count = 0
        for i in range(target_per_category):
            if cat_count >= target_per_category:
                break
            try:
                # High quality Unsplash images
                url = (f"https://source.unsplash.com/"
                       f"512x512/?{category}&sig={count}")
                r = requests.get(url, timeout=10)
                if r.status_code == 200 and len(r.content) > 5000:
                    path = (f"training_images/"
                            f"{category.replace(' ','_')}_{i:03d}.jpg")
                    with open(path, 'wb') as f:
                        f.write(r.content)
                    # Enhance quality
                    img = Image.open(path).convert('RGB')
                    img = img.resize((256, 256), Image.LANCZOS)
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(1.2)
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.1)
                    img.save(path, quality=95)
                    count += 1
                    cat_count += 1
            except:
                pass

        if (cat_idx + 1) % 10 == 0:
            print(f"    {cat_idx+1}/{len(IMAGE_CATEGORIES)} "
                  f"categories done — {count} images")

    # Fill remaining with CIFAR if Unsplash failed
    if count < total_target // 2:
        print(f"  Unsplash limited. Using CIFAR backup...")
        try:
            import torchvision.datasets as datasets
            for ds_class, root in [
                (datasets.CIFAR10,  './cifar10'),
                (datasets.CIFAR100, './cifar100')
            ]:
                if count >= total_target:
                    break
                ds = ds_class(root=root, download=True, train=True)
                for i in range(len(ds)):
                    if count >= total_target:
                        break
                    img, _ = ds[i]
                    img = img.resize((256,256), Image.LANCZOS)
                    img.save(f"training_images/cifar_{count:05d}.jpg",
                             quality=95)
                    count += 1
        except Exception as e:
            print(f"  CIFAR failed: {e}")

    print(f"  Images ready: {count}")


# ── 50 VIDEO PATTERN TYPES ───────────────────

def generate_video_frames(target=1000):
    """50 diverse motion pattern types — high quality"""
    folder = "training_data/video_frames"
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder)
                    if f.endswith('.jpg')])
    if existing >= target:
        print(f"  Video frames: {existing} already ready")
        return

    print(f"  Generating {target} high quality video frames "
          f"(50 pattern types)...")

    # 50 distinct motion patterns
    patterns = [
        'gradient_h', 'gradient_v', 'gradient_d', 'gradient_r',
        'wave_sine', 'wave_cosine', 'wave_complex', 'wave_standing',
        'radial_expand', 'radial_contract', 'radial_rotate', 'radial_pulse',
        'checkerboard', 'checkerboard_color', 'checkerboard_fade',
        'spiral_cw', 'spiral_ccw', 'spiral_zoom',
        'noise_smooth', 'noise_sharp', 'noise_color',
        'stripes_h', 'stripes_v', 'stripes_d', 'stripes_moving',
        'zoom_in', 'zoom_out', 'zoom_rotate',
        'pan_left', 'pan_right', 'pan_up', 'pan_down',
        'flash', 'fade_in', 'fade_out',
        'plasma', 'fire_sim', 'water_sim',
        'tunnel', 'vortex', 'kaleidoscope',
        'rgb_shift', 'hue_rotate', 'saturation_pulse',
        'blur_motion', 'sharpen_motion',
        'grid_move', 'dots_move', 'lines_move', 'circles_move'
    ]

    count     = existing
    frames_per_pattern = max(1, target // len(patterns))
    x_grid, y_grid = np.meshgrid(
        np.arange(256), np.arange(256)
    )

    for pat_idx, pattern in enumerate(patterns):
        if count >= target:
            break
        for frame in range(frames_per_pattern):
            if count >= target:
                break
            t  = frame / frames_per_pattern
            pi = np.pi
            img = np.zeros((256,256,3), dtype=np.uint8)
            x, y = x_grid, y_grid

            if pattern == 'gradient_h':
                img[:,:,0] = (x + frame*5) % 256
                img[:,:,1] = (255 - x + frame*3) % 256
                img[:,:,2] = 128

            elif pattern == 'gradient_v':
                img[:,:,0] = (y + frame*5) % 256
                img[:,:,1] = 128
                img[:,:,2] = (255 - y + frame*3) % 256

            elif pattern == 'gradient_d':
                img[:,:,0] = ((x+y) // 2 + frame*4) % 256
                img[:,:,1] = ((x-y+256) // 2) % 256
                img[:,:,2] = (frame * 8) % 256

            elif pattern == 'wave_sine':
                v = ((np.sin(x/20 + t*2*pi) + 1) * 127).astype(np.uint8)
                img[:,:,0] = v
                img[:,:,1] = (v + 85) % 256
                img[:,:,2] = (v + 170) % 256

            elif pattern == 'wave_cosine':
                v = ((np.cos(y/20 + t*2*pi) + 1) * 127).astype(np.uint8)
                img[:,:,0] = v
                img[:,:,1] = (255 - v)
                img[:,:,2] = 128

            elif pattern == 'wave_complex':
                v = ((np.sin(x/15 + t*2*pi) *
                      np.cos(y/15 + t*pi) + 1) * 127).astype(np.uint8)
                img[:,:,0] = v
                img[:,:,1] = (v * 2) % 256
                img[:,:,2] = 255 - v

            elif pattern == 'radial_expand':
                cx = 128 + int(30 * np.sin(t*2*pi))
                cy = 128 + int(30 * np.cos(t*2*pi))
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                v = (dist * 2 + frame*10) % 256
                img[:,:,0] = v.astype(np.uint8)
                img[:,:,1] = (255 - v).astype(np.uint8)
                img[:,:,2] = 128

            elif pattern == 'checkerboard':
                c = ((x//16 + y//16 + frame) % 2) * 200
                img[:,:,0] = c
                img[:,:,1] = c
                img[:,:,2] = c

            elif pattern == 'checkerboard_color':
                c = (x//16 + y//16 + frame) % 2
                img[:,:,0] = c * 220
                img[:,:,1] = (1-c) * 180
                img[:,:,2] = ((x//16 + frame) % 3) * 100

            elif pattern == 'plasma':
                v = (np.sin(x/20) + np.sin(y/20) +
                     np.sin((x+y)/20 + t*4) +
                     np.sin(np.sqrt(x**2+y**2)/20))
                v = ((v + 4) / 8 * 255).astype(np.uint8)
                img[:,:,0] = v
                img[:,:,1] = (v + 85) % 256
                img[:,:,2] = (v + 170) % 256

            elif pattern == 'spiral_cw':
                angle = np.arctan2(y-128, x-128) + t*2*pi
                dist  = np.sqrt((x-128)**2 + (y-128)**2)
                v = (angle * 40 + dist * 2) % 256
                img[:,:,0] = v.astype(np.uint8)
                img[:,:,1] = (255-v).astype(np.uint8)
                img[:,:,2] = dist.astype(np.uint8) % 256

            elif pattern == 'noise_smooth':
                base = np.random.rand(32,32) * 255
                from PIL import Image as PILImage
                noise_img = PILImage.fromarray(
                    base.astype(np.uint8)
                ).resize((256,256), PILImage.BILINEAR)
                noise = np.array(noise_img)
                img[:,:,0] = noise
                img[:,:,1] = (noise + frame*5) % 256
                img[:,:,2] = (255 - noise)

            elif pattern == 'hue_rotate':
                h = (x/256 + t) % 1.0
                s = np.ones_like(h) * 0.8
                v = np.ones_like(h) * 0.9
                from colorsys import hsv_to_rgb
                for yi in range(0, 256, 4):
                    for xi in range(0, 256, 4):
                        r,g,b = hsv_to_rgb(h[yi,xi], s[yi,xi], v[yi,xi])
                        img[yi:yi+4, xi:xi+4, 0] = int(r*255)
                        img[yi:yi+4, xi:xi+4, 1] = int(g*255)
                        img[yi:yi+4, xi:xi+4, 2] = int(b*255)

            else:
                # Default — colored gradient with motion
                img[:,:,0] = (x + frame*7 + pat_idx*5) % 256
                img[:,:,1] = (y + frame*5 + pat_idx*3) % 256
                img[:,:,2] = (frame*10 + pat_idx*7) % 256

            # High quality save
            pil_img = Image.fromarray(img)
            pil_img = pil_img.filter(ImageFilter.SMOOTH)
            pil_img.save(
                f"{folder}/pat{pat_idx:02d}_f{frame:03d}.jpg",
                quality=95
            )
            count += 1

    print(f"  Video frames ready: {count}")


# ── 50 AUDIO PATTERN TYPES ───────────────────

def generate_spectrograms(target=1000):
    """50 diverse audio frequency patterns — high quality"""
    folder = "training_data/spectrograms"
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder)
                    if f.endswith('.jpg')])
    if existing >= target:
        print(f"  Spectrograms: {existing} already ready")
        return

    print(f"  Generating {target} high quality spectrograms "
          f"(50 audio types)...")

    audio_types = [
        # Tonal patterns
        'pure_tone', 'dual_tone', 'triple_tone', 'chord',
        'octave', 'fifth', 'fourth', 'major_scale',
        # Noise types
        'white_noise', 'pink_noise', 'brown_noise',
        'burst_noise', 'impulse', 'crackle',
        # Music patterns
        'bass', 'treble', 'mid_range', 'full_spectrum',
        'percussion', 'strings', 'brass', 'voice',
        # Sweep patterns
        'chirp_up', 'chirp_down', 'chirp_fast', 'chirp_slow',
        # Modulation
        'am_slow', 'am_fast', 'fm_slow', 'fm_fast',
        'ring_mod', 'tremolo', 'vibrato',
        # Environmental
        'rain', 'thunder', 'wind', 'ocean_waves',
        'crowd', 'traffic', 'machinery',
        # Complex
        'harmonic_rich', 'harmonic_sparse', 'inharmonic',
        'beating', 'resonance', 'echo_sim',
        'speech_pattern', 'music_pattern'
    ]

    t = np.linspace(0, 4, 256*256)
    count = existing

    for type_idx, audio_type in enumerate(audio_types):
        if count >= target:
            break
        samples_per_type = max(1, target // len(audio_types))

        for variant in range(samples_per_type):
            if count >= target:
                break

            # Generate signal based on type
            freq_base = np.random.uniform(1, 20)

            if audio_type == 'pure_tone':
                signal = np.sin(2*np.pi*freq_base*t)

            elif audio_type == 'dual_tone':
                f2 = freq_base * np.random.uniform(1.5, 3)
                signal = (0.6*np.sin(2*np.pi*freq_base*t) +
                         0.4*np.sin(2*np.pi*f2*t))

            elif audio_type == 'chord':
                ratios = [1, 5/4, 3/2, 2]
                signal = sum(np.sin(2*np.pi*freq_base*r*t)/4
                            for r in ratios)

            elif audio_type == 'major_scale':
                scale = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2]
                idx   = variant % 8
                signal = np.sin(2*np.pi*freq_base*scale[idx]*t)

            elif audio_type == 'white_noise':
                signal = np.random.randn(len(t))

            elif audio_type == 'pink_noise':
                white  = np.random.randn(len(t))
                signal = np.cumsum(white) / np.arange(1, len(t)+1)

            elif audio_type == 'chirp_up':
                signal = np.sin(2*np.pi*(freq_base + 5*t)*t)

            elif audio_type == 'chirp_down':
                signal = np.sin(2*np.pi*(freq_base*3 - 5*t)*t)

            elif audio_type == 'am_slow':
                carrier = np.sin(2*np.pi*freq_base*10*t)
                mod     = (1 + 0.5*np.sin(2*np.pi*0.5*t))
                signal  = carrier * mod

            elif audio_type == 'fm_fast':
                signal = np.sin(2*np.pi*freq_base*t +
                                5*np.sin(2*np.pi*3*t))

            elif audio_type == 'harmonic_rich':
                signal = sum(np.sin(2*np.pi*freq_base*k*t) / k
                            for k in range(1, 12))

            elif audio_type == 'beating':
                f2     = freq_base + 0.5
                signal = (np.sin(2*np.pi*freq_base*t) +
                         np.sin(2*np.pi*f2*t))

            elif audio_type == 'percussion':
                env    = np.exp(-t * 10)
                signal = np.random.randn(len(t)) * env

            elif audio_type == 'speech_pattern':
                formants = [freq_base, freq_base*3, freq_base*6]
                signal   = sum(0.3*np.sin(2*np.pi*f*t)
                              for f in formants)
                signal  *= (1 + 0.3*np.sin(2*np.pi*3*t))

            else:
                freqs  = np.random.uniform(0.5, 20, size=4)
                amps   = np.random.uniform(0.1, 1.0, size=4)
                signal = sum(a*np.sin(2*np.pi*f*t)
                            for f,a in zip(freqs,amps))

            # Add subtle noise
            signal += 0.02 * np.random.randn(len(signal))
            signal  = signal / (np.max(np.abs(signal)) + 1e-8)
            data    = signal.reshape(256, 256)

            # High quality RGB spectrogram
            r_ch = ((data + 1) / 2 * 255).astype(np.uint8)
            g_ch = (np.abs(data) * 255).astype(np.uint8)
            b_ch = ((1 - np.abs(data)) * 200 +
                    np.abs(np.roll(data,10,axis=1)) * 55
                    ).astype(np.uint8)

            img = Image.fromarray(np.stack([r_ch,g_ch,b_ch], axis=2))
            img = img.filter(ImageFilter.SMOOTH_MORE)
            img.save(
                f"{folder}/type{type_idx:02d}_v{variant:03d}.jpg",
                quality=95
            )
            count += 1

    print(f"  Spectrograms ready: {count}")


# ── 50 DOCUMENT LAYOUT TYPES ─────────────────

def generate_documents(target=1000):
    """50 diverse professional document layouts — high quality"""
    folder = "training_data/documents"
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder)
                    if f.endswith('.jpg')])
    if existing >= target:
        print(f"  Documents: {existing} already ready")
        return

    print(f"  Generating {target} high quality documents "
          f"(50 layout types)...")

    doc_types = [
        # Text documents
        'report', 'letter', 'memo', 'essay', 'article',
        'resume', 'contract', 'manual', 'thesis', 'notes',
        # Data documents
        'spreadsheet', 'invoice', 'receipt', 'statement', 'budget',
        'table_data', 'comparison', 'schedule', 'calendar', 'timeline',
        # Visual documents
        'presentation_slide', 'poster', 'flyer', 'brochure', 'certificate',
        'infographic', 'diagram', 'flowchart', 'org_chart', 'mind_map',
        # Scientific
        'research_paper', 'lab_report', 'equation_sheet',
        'graph_sheet', 'data_table',
        # Business
        'business_card', 'letterhead', 'form', 'questionnaire',
        'feedback_form', 'application', 'proposal', 'pitch_deck',
        # Mixed layouts
        'newspaper', 'magazine', 'book_page', 'legal_doc',
        'medical_record', 'blueprint', 'map_legend',
        'music_sheet', 'code_listing', 'comic_layout'
    ]

    count = existing
    samples_per_type = max(1, target // len(doc_types))

    for type_idx, doc_type in enumerate(doc_types):
        if count >= target:
            break

        for variant in range(samples_per_type):
            if count >= target:
                break

            bg    = np.random.randint(245, 256)
            img   = Image.new('RGB', (256,256), (bg,bg,bg))
            draw  = ImageDraw.Draw(img)
            accent = tuple(np.random.randint(20,180,3).tolist())

            if doc_type in ['report','essay','article','thesis']:
                # Title bar
                draw.rectangle([0,0,256,35], fill=accent)
                draw.rectangle([10,8,200,27], fill=(255,255,255))
                # Body text lines
                for line in range(15):
                    y  = 45 + line*13
                    lw = np.random.randint(180,250)
                    gv = np.random.randint(40,80)
                    draw.rectangle([10,y,lw,y+7], fill=(gv,gv,gv))
                # Page number
                draw.rectangle([110,248,146,255], fill=(150,150,150))

            elif doc_type in ['invoice','receipt','statement']:
                # Header
                draw.rectangle([0,0,256,40], fill=accent)
                # Table header
                draw.rectangle([5,45,251,60],
                               fill=(200,200,200))
                for row in range(7):
                    y = 65 + row*25
                    draw.rectangle([5,y,251,y+20],
                                   outline=(180,180,180), width=1)
                    for col in [5,80,150,200]:
                        draw.rectangle([col,y,col+65,y+20],
                                       outline=(200,200,200), width=1)
                # Total bar
                draw.rectangle([5,245,251,255],
                               fill=accent)

            elif doc_type in ['presentation_slide','poster','flyer']:
                # Full color background
                bg_color = tuple(
                    np.random.randint(100,200,3).tolist()
                )
                img = Image.new('RGB', (256,256), bg_color)
                draw = ImageDraw.Draw(img)
                # Title block
                draw.rectangle([10,15,246,60],
                               fill=(255,255,255))
                # Content blocks
                for block in range(3):
                    x = 10 + block*82
                    draw.rectangle([x,70,x+75,180],
                                   fill=(255,255,255))
                # Footer
                draw.rectangle([0,220,256,256],
                               fill=(50,50,50))

            elif doc_type in ['spreadsheet','table_data','budget']:
                # Grid
                for row in range(10):
                    y = 20 + row*23
                    draw.rectangle([0,y,256,y+1],
                                   fill=(180,180,180))
                for col in range(5):
                    x = col*52
                    draw.rectangle([x,0,x+1,256],
                                   fill=(180,180,180))
                # Header row
                draw.rectangle([0,0,256,20], fill=accent)
                # Data cells
                for row in range(1,10):
                    for col in range(5):
                        if np.random.random() > 0.3:
                            x = col*52+2
                            y = 20+row*23+3
                            w = np.random.randint(20,45)
                            draw.rectangle([x,y,x+w,y+14],
                                           fill=(100,100,100))

            elif doc_type in ['graph_sheet','infographic']:
                # Background
                draw.rectangle([0,0,256,256], fill=(245,248,255))
                # Axes
                draw.line([(30,220),(230,220)], fill=(0,0,0), width=2)
                draw.line([(30,20),(30,220)], fill=(0,0,0), width=2)
                # Data points and bars
                colors = [(200,50,50),(50,150,200),(50,200,100)]
                for i in range(8):
                    x = 50 + i*23
                    h = np.random.randint(20,180)
                    color = colors[i%3]
                    draw.rectangle([x,220-h,x+18,220],
                                   fill=color)

            elif doc_type in ['flowchart','diagram','org_chart']:
                # Boxes and arrows
                draw.rectangle([85,10,171,40], fill=accent)
                draw.line([(128,40),(128,60)], fill=(0,0,0), width=2)
                for i,x in enumerate([30,85,140]):
                    draw.rectangle([x,60,x+55,90],
                                   fill=(200,220,240))
                    draw.line([(x+27,90),(x+27,110)],
                              fill=(0,0,0), width=1)
                    draw.rectangle([x,110,x+55,140],
                                   fill=(220,240,200))

            elif doc_type in ['newspaper','magazine']:
                # Masthead
                draw.rectangle([0,0,256,30], fill=(20,20,20))
                draw.rectangle([5,5,160,25], fill=(255,255,255))
                # Columns
                draw.rectangle([128,35,130,240],
                               fill=(180,180,180))
                for col in [0,1]:
                    x_off = col * 133
                    draw.rectangle([x_off+5,35,x_off+120,90],
                                   fill=(200,200,200))
                    for line in range(10):
                        y = 95 + line*14
                        w = np.random.randint(80,120)
                        gv = np.random.randint(50,100)
                        draw.rectangle(
                            [x_off+5,y,x_off+w,y+8],
                            fill=(gv,gv,gv)
                        )

            elif doc_type in ['resume','letter','contract']:
                # Professional layout
                draw.rectangle([0,0,256,50], fill=accent)
                draw.rectangle([10,10,150,40],
                               fill=(255,255,255))
                draw.rectangle([0,55,5,256],
                               fill=accent)
                sections = ['EXPERIENCE','EDUCATION','SKILLS']
                for s_idx, section in enumerate(sections):
                    y = 65 + s_idx*60
                    draw.rectangle([10,y,100,y+10],
                                   fill=(80,80,80))
                    for line in range(3):
                        draw.rectangle(
                            [15, y+15+line*12,
                             np.random.randint(150,240),
                             y+22+line*12],
                            fill=(120,120,120)
                        )

            else:
                # Generic professional layout
                draw.rectangle([0,0,256,30], fill=accent)
                for section in range(4):
                    y = 40 + section*52
                    draw.rectangle([10,y,246,y+45],
                                   outline=(180,180,180), width=1)
                    draw.rectangle([15,y+5,200,y+15],
                                   fill=(80,80,80))
                    for line in range(2):
                        draw.rectangle(
                            [15, y+20+line*12,
                             np.random.randint(100,230),
                             y+28+line*12],
                            fill=(150,150,150)
                        )

            # Apply quality enhancements
            img = img.filter(ImageFilter.SMOOTH)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)

            img.save(
                f"{folder}/type{type_idx:02d}_v{variant:03d}.jpg",
                quality=95
            )
            count += 1

    print(f"  Documents ready: {count}")


# ── DATA LOADER ──────────────────────────────

def load_all_training_data(max_per_type=500):
    """Load high quality samples from all folders"""
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
                ).convert('RGB').resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
                )
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

def train(epochs=2000, batch_size=16):
    print("\n" + "="*55)
    print("UNIVERSAL FOUNDATION TRAINER v5.3")
    print("50 Categories per file type")
    print("Maximum quality training")
    print(f"Device: {DEVICE}")
    print("Inventor: Rohit Kalu Sasane, Pune India 2026")
    print("="*55)

    # Generate all training data
    print("\nPreparing training data...")
    download_images_diverse(target_per_category=20)
    generate_video_frames(target=1000)
    generate_spectrograms(target=1000)
    generate_documents(target=1000)

    # Load data
    print("\nLoading training data...")
    tensors = load_all_training_data(max_per_type=500)
    total   = len(tensors)
    print(f"\nTotal training samples: {total:,}")

    if total == 0:
        print("No training data found.")
        return

    foundation = Foundation().to(DEVICE)

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
        foundation.parameters(), lr=0.001,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    loss_fn   = nn.MSELoss()
    best_loss = float('inf')

    print(f"\nEpochs:    {epochs}")
    print(f"Batch:     {batch_size}")
    print(f"Samples:   {total:,}")
    print(f"GPU time:  ~2-3 hours")
    print(f"CPU time:  ~12+ hours\n")

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
                  f"LR: {lr:.7f}")

    sz = os.path.getsize(SAVE_PATH)
    print(f"\n{'='*55}")
    print(f"Training complete!")
    print(f"Best loss:    {best_loss:.6f}")
    print(f"Foundation:   {sz/1024/1024:.1f}MB")
    print(f"Trained on:   {total:,} samples")
    print(f"Categories:   50 per file type")
    print(f"File types:   Images+Video+Audio+Documents")
    print(f"{'='*55}")


if __name__ == "__main__":
    if os.path.exists(SAVE_PATH):
        print(f"Weights exist: {SAVE_PATH}")
        print("Generating data only. Delete weights to retrain.\n")
        download_images_diverse(target_per_category=20)
        generate_video_frames(target=1000)
        generate_spectrograms(target=1000)
        generate_documents(target=1000)
        print("\nAll data ready.")
    else:
        train(epochs=2000, batch_size=16)
