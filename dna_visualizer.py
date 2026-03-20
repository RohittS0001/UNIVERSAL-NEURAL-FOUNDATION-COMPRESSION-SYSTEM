import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import os
import struct
import hashlib
import base64

# ============================================
# DNA VISUALIZER v4
# Works with v4 Hierarchical Foundation
# Shows DNA as copyable string
# Reconstruct from DNA string anywhere
# Perfect for demos and pitching
# Inventor: Rohit Kalu Sasane, Pune India 2026
# ============================================

# ── V4 ARCHITECTURE ───────────────────────────
class GlobalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(128 * 4 * 4, 32)
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
            nn.Linear(256 * 4 * 4, 256)
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
            nn.Flatten(),
            nn.Linear(512 * 16 * 16, 512)
        )
    def forward(self, x): return self.net(x)

class HierarchicalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.combine = nn.Sequential(
            nn.Linear(800, 512 * 16 * 16), nn.ReLU()
        )
        self.decode = nn.Sequential(
            nn.Unflatten(1, (512, 16, 16)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, g, r, d):
        return self.decode(self.combine(torch.cat([g, r, d], dim=1)))

class HierarchicalFoundation(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_encoder = GlobalEncoder()
        self.regional_encoder = RegionalEncoder()
        self.detail_encoder = DetailEncoder()
        self.decoder = HierarchicalDecoder()

    def encode(self, x):
        with torch.no_grad():
            g = self.global_encoder(x)
            r = self.regional_encoder(x)
            d = self.detail_encoder(x)
        return g, r, d

    def decode(self, g, r, d):
        with torch.no_grad():
            return self.decoder(g, r, d)


# ── LOAD FOUNDATION ───────────────────────────
def load_foundation():
    foundation = HierarchicalFoundation()
    weights_path = "foundation_v4_weights.pth"

    if not os.path.exists(weights_path):
        print("No v4 foundation found.")
        print("Run train_foundation.py first.")
        exit()

    checkpoint = torch.load(weights_path, weights_only=True, map_location='cpu')
    foundation_dict = foundation.state_dict()
    
    # Filter compatible weights only
    compatible_dict = {}
    for k, v in checkpoint.items():
        if k in foundation_dict and v.shape == foundation_dict[k].shape:
            compatible_dict[k] = v
    
    num_loaded = len(compatible_dict)
    num_total = len(checkpoint)
    print(f"Compatible weights: {num_loaded}/{num_total}")
    
    foundation.load_state_dict(compatible_dict, strict=False)
    foundation.eval()
    
    size = os.path.getsize(weights_path)
    print(f"Foundation loaded: {size/1024/1024:.1f} MB")
    return foundation



# ── DNA STRING FORMAT ─────────────────────────
# Encode all three chains as one base64 string
# Global(32) + Regional(256) + Detail(512) = 800 floats

def chains_to_dna_string(g, r, d):
    """Convert three chains to one copyable DNA string"""
    g_arr = g.numpy().flatten().astype(np.float32)
    r_arr = r.numpy().flatten().astype(np.float32)
    d_arr = d.numpy().flatten().astype(np.float32)

    # Pack all three chains with separator markers
    data = struct.pack('HHH', len(g_arr), len(r_arr), len(d_arr))
    data += g_arr.tobytes()
    data += r_arr.tobytes()
    data += d_arr.tobytes()

    return base64.b64encode(data).decode('utf-8')


def dna_string_to_chains(dna_string):
    """Decode DNA string back to three chains"""
    data = base64.b64decode(dna_string)
    g_dim, r_dim, d_dim = struct.unpack('HHH', data[:6])

    offset = 6
    g = np.frombuffer(data[offset:offset+g_dim*4], dtype=np.float32).copy()
    offset += g_dim * 4
    r = np.frombuffer(data[offset:offset+r_dim*4], dtype=np.float32).copy()
    offset += r_dim * 4
    d = np.frombuffer(data[offset:offset+d_dim*4], dtype=np.float32).copy()

    return (
        torch.FloatTensor(g).unsqueeze(0),
        torch.FloatTensor(r).unsqueeze(0),
        torch.FloatTensor(d).unsqueeze(0)
    )


# ── MAIN FUNCTIONS ────────────────────────────

def image_to_dna(image_path):
    """Convert any image to a copyable DNA string"""
    foundation = load_foundation()

    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    tensor = torch.FloatTensor(
        np.array(img.resize((256, 256))) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)

    g, r, d = foundation.encode(tensor)
    reconstructed = foundation.decode(g, r, d)
    loss = nn.MSELoss()(reconstructed, tensor)

    dna_string = chains_to_dna_string(g, r, d)

    # Save DNA text file
    dna_path = image_path + ".dna.txt"
    file_size = os.path.getsize(image_path)
    with open(dna_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("UNIVERSAL NEURAL FOUNDATION v4 — DNA FILE\n")
        f.write("Invented by Rohit Kalu Sasane, Pune India 2026\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Original file:       {os.path.basename(image_path)}\n")
        f.write(f"Original size:       {file_size:,} bytes\n")
        f.write(f"Original dimensions: {original_size[0]}x{original_size[1]}\n")
        f.write(f"DNA size:            {len(dna_string)} characters\n")
        f.write(f"DNA chains:          Global(32) + Regional(256) + Detail(512)\n")
        f.write(f"Reconstruction loss: {loss.item():.6f}\n")
        f.write(f"Checksum:            {hashlib.md5(open(image_path,'rb').read()).hexdigest()}\n\n")
        f.write("DNA STRING — copy everything between the lines:\n")
        f.write("-" * 60 + "\n")
        f.write(dna_string + "\n")
        f.write("-" * 60 + "\n\n")
        f.write("To reconstruct: run dna_visualizer.py option 2\n")
        f.write("Paste the DNA string when prompted.\n")

    print("\n" + "=" * 60)
    print("DNA GENERATED")
    print("=" * 60)
    print(f"Original:    {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    print(f"DNA string:  {len(dna_string)} characters")
    print(f"Compression: {(1-len(dna_string)/file_size)*100:.1f}% smaller")
    print(f"Loss:        {loss.item():.6f}")
    print(f"\nDNA PREVIEW (first 80 chars):")
    print(f"{dna_string[:80]}...")
    print(f"\nFull DNA saved to: {dna_path}")
    print("=" * 60)

    return dna_string, dna_path


def dna_to_image(dna_string, output_path="reconstructed_from_dna.png"):
    """Reconstruct any image from DNA string alone"""
    print("\n" + "=" * 60)
    print("RECONSTRUCTING FROM DNA")
    print("=" * 60)

    foundation = load_foundation()

    try:
        g, r, d = dna_string_to_chains(dna_string)
        print(f"DNA decoded: {g.shape[1]} + {r.shape[1]} + {d.shape[1]} = 800 values")
    except Exception as e:
        print(f"Invalid DNA string: {e}")
        return None

    reconstructed = foundation.decode(g, r, d)
    final = torch.clamp(reconstructed, 0, 1)
    img_array = (
        final.squeeze(0).permute(1,2,0).numpy() * 255
    ).astype(np.uint8)

    Image.fromarray(img_array).save(output_path)

    print(f"Reconstructed: {output_path}")
    print("No original file needed.")
    print("Foundation + DNA string = Image.")
    print("=" * 60)

    return output_path


def create_pitch_card(image_path, dna_string):
    """Create professional pitch card for investors"""
    print("\nCreating pitch card...")

    if not os.path.exists("reconstructed_from_dna.png"):
        print("Run option 2 first to create reconstruction.")
        return

    original = Image.open(image_path).convert('RGB')
    reconstructed = Image.open("reconstructed_from_dna.png").convert('RGB')

    orig_size = os.path.getsize(image_path)
    dna_size = len(dna_string)
    compression = (1 - dna_size/orig_size) * 100

    card_w, card_h = 1200, 720
    card = Image.new('RGB', (card_w, card_h), (12, 12, 20))
    draw = ImageDraw.Draw(card)

    # Images
    img_size = (480, 480)
    orig_resized = original.resize(img_size)
    recon_resized = reconstructed.resize(img_size)

    card.paste(orig_resized, (40, 160))
    card.paste(recon_resized, (680, 160))

    # Title
    draw.text((600, 20), "UNIVERSAL NEURAL FOUNDATION",
              fill=(100, 200, 255), anchor="mt")
    draw.text((600, 55), "Invented by Rohit Kalu Sasane — Pune, India — 2026",
              fill=(150, 150, 150), anchor="mt")
    draw.text((600, 88), "Patent Pending — 13 Claims Filed",
              fill=(100, 180, 100), anchor="mt")

    # Labels
    draw.text((280, 130), f"ORIGINAL — {orig_size/1024/1024:.1f} MB",
              fill=(255, 255, 255), anchor="mt")
    draw.text((920, 130), f"DNA RECONSTRUCTION — {dna_size} chars",
              fill=(100, 255, 100), anchor="mt")

    # Arrow and DNA
    draw.text((600, 340), "→", fill=(255, 200, 0), anchor="mm")
    draw.text((600, 380), "DNA", fill=(255, 200, 0), anchor="mm")
    draw.text((600, 415), dna_string[:50] + "...",
              fill=(180, 180, 80), anchor="mm")

    # Stats bar
    draw.rectangle([(0, 650), (card_w, 720)], fill=(20, 20, 35))
    draw.text((600, 668),
              f"{orig_size/1024/1024:.1f} MB compressed to {dna_size} characters — {compression:.1f}% smaller — 3D Hierarchical DNA — Patent Pending",
              fill=(100, 255, 100), anchor="mt")
    draw.text((600, 698),
              "Foundation + DNA = Reconstruction — No original file needed — Rohit Sasane 2026",
              fill=(120, 120, 120), anchor="mt")

    card.save("pitch_card.png")
    print("Pitch card saved: pitch_card.png")
    print("Show this to investors, professors, and data center CTOs.")


# ── MAIN ──────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("UNIVERSAL NEURAL FOUNDATION v4 — DNA VISUALIZER")
    print("Invented by Rohit Kalu Sasane, Pune India 2026")
    print("=" * 60)
    print("\n1. Convert image to DNA string")
    print("2. Reconstruct image from DNA string")
    print("3. Full demo — DNA + reconstruct + pitch card")
    print()

    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        image_file = input("Image filename (e.g. demo.png): ").strip()
        if os.path.exists(image_file):
            dna_string, dna_path = image_to_dna(image_file)
            print(f"\nOpen {dna_path} and copy the DNA string.")
        else:
            print(f"File not found: {image_file}")

    elif choice == "2":
        print("Paste your DNA string and press Enter:")
        dna_input = input().strip()
        output = input("Output filename (press Enter for default): ").strip()
        if not output:
            output = "reconstructed_from_dna.png"
        dna_to_image(dna_input, output)

    elif choice == "3":
        image_file = input("Image filename (e.g. demo.png): ").strip()
        if os.path.exists(image_file):
            print("\nStep 1: Converting to DNA...")
            dna_string, dna_path = image_to_dna(image_file)
            print("\nStep 2: Reconstructing from DNA...")
            dna_to_image(dna_string)
            print("\nStep 3: Creating pitch card...")
            create_pitch_card(image_file, dna_string)
            print("\n" + "="*60)
            print("COMPLETE")
            print("="*60)
            print(f"DNA file:    {dna_path}")
            print(f"Pitch card:  pitch_card.png")
            print("Ready to demo to anyone.")
        else:
            print(f"File not found: {image_file}")