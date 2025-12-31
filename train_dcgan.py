import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageGlobDataset(Dataset):
    def __init__(self, root: str, image_size: int = 64):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"data dir not found: {self.root}")

        # 再帰的に画像を拾う（imagefolderの構造でも素のフォルダでもOK）
        self.paths = [p for p in self.root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under: {self.root}")

        self.tfm = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm(img)


# DCGAN (64x64)
class Generator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            # (nz) -> 4x4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).view(-1)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="path to extracted images folder")
    ap.add_argument("--outdir", type=str, default="runs/dcgan_ffhq64")
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--nz", type=int, default=128)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--ndf", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="mixed precision")
    ap.add_argument("--save_every", type=int, default=500)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "ckpt"), exist_ok=True)

    ds = ImageGlobDataset(args.data_dir, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)

    G = Generator(nz=args.nz, ngf=args.ngf).to(device)
    D = Discriminator(ndf=args.ndf).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    fixed_z = torch.randn(64, args.nz, 1, 1, device=device)

    step = 0
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}")
        for real in pbar:
            real = real.to(device, non_blocking=True)
            bs = real.size(0)

            # --- Train D ---
            z = torch.randn(bs, args.nz, 1, 1, device=device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                fake = G(z).detach()
                D_real = D(real)
                D_fake = D(fake)

                y_real = torch.ones_like(D_real)
                y_fake = torch.zeros_like(D_fake)

                lossD = bce(D_real, y_real) + bce(D_fake, y_fake)

            optD.zero_grad(set_to_none=True)
            scaler.scale(lossD).backward()
            scaler.step(optD)

            # --- Train G ---
            z = torch.randn(bs, args.nz, 1, 1, device=device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                fake = G(z)
                D_fake2 = D(fake)
                y = torch.ones_like(D_fake2)
                lossG = bce(D_fake2, y)

            optG.zero_grad(set_to_none=True)
            scaler.scale(lossG).backward()
            scaler.step(optG)
            scaler.update()

            if step % 50 == 0:
                pbar.set_postfix(lossD=float(lossD.item()), lossG=float(lossG.item()))

            if step % args.save_every == 0:
                G.eval()
                with torch.no_grad():
                    sample = G(fixed_z).cpu()
                G.train()
                save_image(sample, os.path.join(args.outdir, "samples", f"step_{step:07d}.png"),
                           nrow=8, normalize=True, value_range=(-1, 1))

                ckpt = {
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "optG": optG.state_dict(),
                    "optD": optD.state_dict(),
                    "step": step,
                    "args": vars(args),
                }
                torch.save(ckpt, os.path.join(args.outdir, "ckpt", f"step_{step:07d}.pt"))

            step += 1

        # epoch end save
        torch.save(G.state_dict(), os.path.join(args.outdir, f"G_epoch{epoch:03d}.pth"))

    print("done")


if __name__ == "__main__":
    main()
