"""Fabrication et signature d'une mise a jour.

Trois commandes:

    python -m invis.release keygen
        Cree la paire de cles. La cle privee reste sur ta machine et ne doit
        jamais entrer dans le depot; la cle publique se colle dans updater.py.

    python -m invis.release package --version 1.1.0
        Fabrique l'archive du code applicatif.

    python -m invis.release sign --version 1.1.0 --key chemin/cle.pem \\
        --base-url https://github.com/OWNER/REPO/releases/download/v1.1.0
        Signe l'archive et ecrit le manifeste a publier.

Le manifeste et l'archive se deposent ensuite sur la page de publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from typing import List

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from invis.version import VERSION
else:
    from .version import VERSION

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = "invis"
OUTPUT_DIR = os.path.join(REPO_ROOT, "dist")

# Ce qui part dans l'archive. Volontairement restrictif: on n'expedie que le
# code, jamais les sessions enregistrees, les cles ou les fichiers temporaires.
INCLUDE_SUFFIXES = (".py",)
EXCLUDE_NAMES = {"__pycache__", "sessions", "test_invis.py", "release.py"}


def collect_files() -> List[str]:
    files = []
    base = os.path.join(REPO_ROOT, PACKAGE_DIR)
    for root, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_NAMES]
        for name in sorted(names):
            if name in EXCLUDE_NAMES or not name.endswith(INCLUDE_SUFFIXES):
                continue
            files.append(os.path.join(root, name))
    return sorted(files)


def archive_path(version: str) -> str:
    return os.path.join(OUTPUT_DIR, f"payload-{version}.zip")


def cmd_keygen(args) -> int:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} existe deja. Utilise --force pour l'ecraser "
              f"(cela invalidera toutes les mises a jour deja publiees).")
        return 1

    private = Ed25519PrivateKey.generate()
    pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        fh.write(pem)
    try:
        os.chmod(args.out, 0o600)
    except OSError:
        pass

    public_hex = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()

    print(f"cle privee ecrite dans {args.out}")
    print("  -> ne la mets JAMAIS dans le depot, et sauvegarde-la:")
    print("     la perdre coupe toute possibilite de publier une mise a jour.")
    print()
    print("Colle cette ligne dans invis/updater.py:")
    print(f'PUBLIC_KEY_HEX = "{public_hex}"')
    return 0


def cmd_package(args) -> int:
    version = args.version or VERSION
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = collect_files()
    if not files:
        print("aucun fichier a empaqueter")
        return 1

    out = archive_path(version)
    # Archive deterministe: meme contenu, meme octets. Deux fabrications
    # successives donnent alors la meme empreinte, ce qui rend une
    # publication verifiable par un tiers.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arc = os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")
            info = zipfile.ZipInfo(arc, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(path, "rb") as fh:
                zf.writestr(info, fh.read())

    size = os.path.getsize(out)
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{out}")
    print(f"  {len(files)} fichiers, {size / 1024:.1f} Ko")
    print(f"  sha256 {digest}")
    return 0


def cmd_sign(args) -> int:
    from cryptography.hazmat.primitives import serialization

    version = args.version or VERSION
    archive = archive_path(version)
    if not os.path.exists(archive):
        print(f"{archive} absent: lance d'abord la commande package")
        return 1

    with open(args.key, "rb") as fh:
        private = serialization.load_pem_private_key(fh.read(), password=None)

    payload = open(archive, "rb").read()
    signature = private.sign(payload).hex()
    digest = hashlib.sha256(payload).hexdigest()

    base = args.base_url.rstrip("/")
    if not base.lower().startswith("https://"):
        print("--base-url doit etre en HTTPS: le programme refuse tout le reste")
        return 1

    manifest = {
        "version": version,
        "url": f"{base}/{os.path.basename(archive)}",
        "sha256": digest,
        "signature": signature,
        "notes": args.notes or "",
    }
    out = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"{out}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print()
    print("A publier ensemble sur la page de release:")
    print(f"  {os.path.basename(archive)}")
    print("  manifest.json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Publication d'une mise a jour signee")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("keygen", help="cree la paire de cles de publication")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "..", "esp32cam-vision-signing.pem"),
                   help="chemin de la cle privee, hors du depot par defaut")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_keygen)

    p = sub.add_parser("package", help="fabrique l'archive du code")
    p.add_argument("--version", default=None)
    p.set_defaults(func=cmd_package)

    p = sub.add_parser("sign", help="signe l'archive et ecrit le manifeste")
    p.add_argument("--version", default=None)
    p.add_argument("--key", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_sign)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
