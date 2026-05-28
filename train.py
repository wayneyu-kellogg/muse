#!/usr/bin/env python3
"""
MUSE Training Script
Multimodal Understanding of Sentiments and Expressions
CNN image encoder + Transformer decoder trained on ArtEmis dataset.

Usage:
    python train.py                            # train from scratch
    python train.py --skip-download            # skip Kaggle download if data exists
    python train.py --resume checkpoints/checkpoint_epoch_5.pt
    python train.py --num-epochs 30 --batch-size 64 --num-layers 3
"""

import argparse
import string
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torchvision
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import transforms
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train MUSE model on ArtEmis')

    # Paths
    parser.add_argument('--root', type=str,
                        default=str(Path(__file__).parent.resolve()),
                        help='Project root directory (default: directory containing this script)')
    parser.add_argument('--kaggle-dataset', type=str,
                        default='rollas/artemis-dataset-including-10k-images')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip Kaggle download (use if data/images already exists)')

    # Training
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader worker processes (use 0 for MPS/local)')

    # Model
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--d-feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-seq-length', type=int, default=40)
    parser.add_argument('--min-freq', type=int, default=2)

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Data download and pre-processing
# ──────────────────────────────────────────────────────────────────────────────

def download_data(data_dir: Path, kaggle_dataset: str):
    """Download and unzip ArtEmis dataset from Kaggle into data_dir."""
    print(f"Downloading {kaggle_dataset} from Kaggle...")
    data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', kaggle_dataset,
         '-p', str(data_dir), '--unzip'],
        check=True
    )
    print("Download complete.")


def presize_images(src_dir: Path, dst_dir: Path, size: int = 256):
    """Resize all JPEGs in src_dir to (size x size) thumbnails in dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = list(src_dir.glob('*.jpg'))
    print(f"Pre-resizing {len(files)} images to {size}×{size}...")
    errors = 0
    for src in tqdm(files, desc='Resizing'):
        dst = dst_dir / src.name
        if dst.exists():
            continue
        try:
            with Image.open(src) as img:
                img = img.convert('RGB')
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                img.save(dst, format='JPEG', quality=85)
        except Exception as e:
            errors += 1
            print(f"  Warning — could not process {src.name}: {e}")
    print(f"Pre-resizing done ({errors} errors).")


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────────────────────

def build_vocab(data: pd.DataFrame, min_freq: int = 2):
    tokenized = data['utterance'].apply(
        lambda x: [w.strip(string.punctuation) for w in str(x).lower().split()]
    )
    word_counts = Counter(w for utt in tokenized for w in utt)
    vocab = {w for w, c in word_counts.items() if c >= min_freq}
    vocab.update({'<PAD>', '<UNK>', '<SOS>', '<EOS>'})
    word2idx = {w: i for i, w in enumerate(sorted(vocab))}
    idx2word = {i: w for w, i in word2idx.items()}
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Special tokens — <PAD>={word2idx['<PAD>']}, <UNK>={word2idx['<UNK>']}, "
          f"<SOS>={word2idx['<SOS>']}, <EOS>={word2idx['<EOS>']}")
    return word2idx, idx2word


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ArtemisDataset(torch.utils.data.Dataset):
    def __init__(self, data, word2idx, img_dir, image_transform, max_seq_length=40):
        self.data = data.reset_index(drop=True)
        self.word2idx = word2idx
        self.img_dir = Path(img_dir)
        self.image_transform = image_transform
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        row = self.data.iloc[key]
        img = Image.open(self.img_dir / row['filename']).convert('RGB')
        img_tensor = self.image_transform(img)

        words = [w.strip(string.punctuation) for w in str(row['utterance']).lower().split()]
        indices = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        indices = [self.word2idx['<SOS>']] + indices + [self.word2idx['<EOS>']]
        if len(indices) < self.max_seq_length:
            indices += [self.word2idx['<PAD>']] * (self.max_seq_length - len(indices))
        else:
            indices = indices[:self.max_seq_length - 1] + [self.word2idx['<EOS>']]

        return img_tensor, torch.tensor(indices)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class CNNImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool + fc → (B,512,7,7)
        self.proj = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.cnn(x)                        # (B, 512, 7, 7)
        x = x.flatten(2).transpose(1, 2)       # (B, 49, 512)
        return self.proj(x)                    # (B, 49, output_dim)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value), attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        q = self.linear_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        out, attn_weights = attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.linear_out(out), attn_weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward, dropout=0.1):
        super().__init__()
        self.self_attention  = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward    = FeedForwardNetwork(d_model, d_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, memory_mask=None):
        T = x.size(1)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        # Masked self-attention
        attn1, _ = self.self_attention(x, x, x, causal_mask)
        x = self.norm1(x + self.dropout1(attn1))

        # Cross-attention over encoder output
        attn2, _ = self.cross_attention(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout2(attn2))

        # Feed-forward
        x = self.norm3(x + self.dropout3(self.feed_forward(x)))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_feedforward=2048,
                 num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding          = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model)
        self.layers             = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, memory_mask=None):
        x = self.embedding(tgt)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, memory, memory_mask)
        return self.fc_out(x)


class MUSE(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_feedforward=2048,
                 num_layers=3, dropout=0.1):
        super().__init__()
        self.encoder = CNNImageEncoder(output_dim=d_model)
        self.decoder = TransformerDecoder(
            vocab_size, d_model, num_heads, d_feedforward, num_layers, dropout
        )

    def forward(self, images, tgt, memory_mask=None):
        memory = self.encoder(images)
        return self.decoder(tgt, memory, memory_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_muse(model, train_loader, val_loader, loss_fn, optimizer, scheduler,
               device, num_epochs, ckpt_dir, start_epoch=0):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        # ── Train ──
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader,
                         desc=f'[Train] Epoch {epoch+1}/{num_epochs}', leave=False)
        for images, tgt in train_bar:
            images, tgt = images.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(images, tgt[:, :-1])
            loss = loss_fn(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train = total_train_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, tgt in val_loader:
                images, tgt = images.to(device), tgt.to(device)
                output = model(images, tgt[:, :-1])
                total_val_loss += loss_fn(
                    output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1)
                ).item()
        avg_val = total_val_loss / len(val_loader)

        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch+1}/{num_epochs}]  '
              f'Train: {avg_train:.4f}  Val: {avg_val:.4f}  LR: {current_lr:.2e}')

        # ── Save checkpoint every epoch ──
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train,
            'val_loss': avg_val,
        }
        torch.save(ckpt, ckpt_dir / f'checkpoint_epoch_{epoch}.pt')

        # ── Save best separately ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(ckpt, ckpt_dir / 'best_checkpoint.pt')
            print(f'  → New best val loss: {best_val_loss:.4f}')


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def generate_caption(model, image, word2idx, idx2word, max_length=40, device='cpu'):
    model.eval()
    image = image.unsqueeze(0).to(device)
    generated = [word2idx['<SOS>']]

    with torch.no_grad():
        for _ in range(max_length):
            tgt = torch.tensor(generated).unsqueeze(0).to(device)
            output = model(image, tgt)
            next_id = torch.argmax(output[0, -1, :]).item()
            generated.append(next_id)
            if next_id == word2idx['<EOS>']:
                break

    return ' '.join(
        idx2word[i] for i in generated[1:]
        if i in idx2word and i != word2idx['<EOS>'] and i != word2idx['<PAD>']
    )


def sample_generated_captions(model, dataset, word2idx, idx2word,
                               num_samples=5, device='cpu', output_file=None):
    """Print sample captions. If output_file is given, also write to disk."""
    model.eval()
    lines = []
    for _ in range(num_samples):
        idx = torch.randint(0, len(dataset), (1,)).item()
        img_tensor, _ = dataset[idx]
        caption = generate_caption(model, img_tensor, word2idx, idx2word, device=device)
        ground_truth = dataset.data.iloc[idx]['utterance']
        lines.append(f"[{idx}] Ground truth : {ground_truth}")
        lines.append(f"[{idx}] Generated    : {caption}")
        lines.append("")

    output = '\n'.join(lines)
    print(output)

    if output_file:
        Path(output_file).write_text(output)
        print(f"Sample captions saved to {output_file}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    ROOT       = Path(args.root)
    DATA_DIR   = ROOT / 'data'
    IMG_DIR    = DATA_DIR / 'images'
    RESIZED_DIR = DATA_DIR / 'images_256'
    CKPT_DIR   = ROOT / 'checkpoints'

    # ── 1. Download data ──────────────────────────────────────────────────────
    if not args.skip_download:
        download_data(DATA_DIR, args.kaggle_dataset)
    else:
        print("Skipping Kaggle download (--skip-download set).")

    # ── 2. Pre-resize images ──────────────────────────────────────────────────
    resized_count = len(list(RESIZED_DIR.glob('*.jpg'))) if RESIZED_DIR.exists() else 0
    if resized_count < 100:
        presize_images(IMG_DIR, RESIZED_DIR)
    else:
        print(f"Found {resized_count} pre-resized images in {RESIZED_DIR}, skipping resize.")

    # ── 3. Load CSV ───────────────────────────────────────────────────────────
    csv_path = DATA_DIR / 'dataset_final.csv'
    data = pd.read_csv(csv_path)
    print(f"Loaded {len(data)} annotations from {csv_path}")

    # ── 4. Vocabulary ─────────────────────────────────────────────────────────
    word2idx, idx2word = build_vocab(data, args.min_freq)
    VOCAB_SIZE = len(word2idx)

    # ── 5. Image transforms ───────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── 6. Train / val / test split ───────────────────────────────────────────
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    train_data, val_data  = train_test_split(train_data, test_size=0.111, random_state=42)
    print(f"Split: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test")

    train_dataset = ArtemisDataset(train_data, word2idx, RESIZED_DIR, train_transform, args.max_seq_length)
    val_dataset   = ArtemisDataset(val_data,   word2idx, RESIZED_DIR, eval_transform,  args.max_seq_length)
    test_dataset  = ArtemisDataset(test_data,  word2idx, RESIZED_DIR, eval_transform,  args.max_seq_length)

    # ── 7. DataLoaders ────────────────────────────────────────────────────────
    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=pin
    )

    # ── 8. Device ─────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # ── 9. Model ──────────────────────────────────────────────────────────────
    model = MUSE(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_feedforward=args.d_feedforward,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    # Freeze ResNet backbone — only train projection layer + decoder
    for param in model.encoder.cnn.parameters():
        param.requires_grad = False
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ── 10. Loss / optimizer / scheduler ─────────────────────────────────────
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=word2idx['<PAD>'],
        label_smoothing=args.label_smoothing
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )

    # ── 11. Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed — starting at epoch {start_epoch + 1}")

    # ── 12. Train ─────────────────────────────────────────────────────────────
    train_muse(
        model, train_loader, val_loader, loss_fn, optimizer, scheduler,
        device, args.num_epochs, CKPT_DIR, start_epoch
    )

    # ── 13. Sample captions from test set ────────────────────────────────────
    print("\n── Sample generated captions (test set) ──")
    sample_generated_captions(
        model, test_dataset, word2idx, idx2word,
        num_samples=10, device=device,
        output_file=str(CKPT_DIR / 'sample_captions.txt')
    )


if __name__ == '__main__':
    main()
