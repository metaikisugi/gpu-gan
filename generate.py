import argparse
import os
import torch
from torchvision.utils import save_image

from train_dcgan import Generator  # 同じフォルダに置いてある前提


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="samples_gen.png")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("args", {})
    nz = int(cfg.get("nz", 128))
    ngf = int(cfg.get("ngf", 64))

    G = Generator(nz=nz, ngf=ngf).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    z = torch.randn(args.n, nz, 1, 1, device=device)
    with torch.no_grad():
        imgs = G(z).cpu()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_image(imgs, args.out, nrow=8, normalize=True, value_range=(-1, 1))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
