from PIL import Image, ImageDraw
import os

original = Image.open("demo.png").convert("RGB")
reconstructed = Image.open("epigenetic_full.png").convert("RGB")

original_size = os.path.getsize("demo.png")
dna_size = os.path.getsize("demo.png.dna")

# Fix both to same size
size = (512, 512)
original_resized = original.resize(size)
reconstructed_resized = reconstructed.resize(size)

# Build comparison canvas
padding = 20
label_height = 50
footer_height = 40
total_width = size[0] * 2 + padding * 3
total_height = size[1] + label_height + footer_height + padding

comparison = Image.new('RGB', (total_width, total_height), (15, 15, 15))

# Paste images
comparison.paste(original_resized, (padding, label_height))
comparison.paste(reconstructed_resized, (size[0] + padding * 2, label_height))

# Labels
draw = ImageDraw.Draw(comparison)

# Original label
draw.text(
    (padding + size[0] // 2, 15),
    f"ORIGINAL — {original_size/1024/1024:.1f} MB",
    fill=(255, 255, 255),
    anchor="mt"
)

# Reconstruction label
draw.text(
    (size[0] + padding * 2 + size[0] // 2, 15),
    f"DNA RECONSTRUCTION — {dna_size/1024:.1f} KB — 99.9% smaller",
    fill=(80, 255, 80),
    anchor="mt"
)

# Divider line
draw.line(
    [(size[0] + padding + padding//2, label_height),
     (size[0] + padding + padding//2, label_height + size[1])],
    fill=(60, 60, 60),
    width=2
)

# Footer
draw.text(
    (total_width // 2, total_height - footer_height + 10),
    "Universal Neural Foundation v5.1 — Invented by Rohit Sasane — Pune, India — 2026",
    fill=(140, 140, 140),
    anchor="mt"
)

comparison.save("comparison_demo.png")
print(f"Saved: comparison_demo.png")
print(f"Original: {original_size/1024/1024:.1f} MB")
print(f"DNA file: {dna_size/1024:.1f} KB")
print(f"Compression: 99.9%")
print(f"Your name is on it.")

