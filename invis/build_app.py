"""Fabrication de l'executable autonome, Windows et Linux.

    python -m invis.build_app

Produit un dossier `dist/EspCamVision/` contenant l'executable et ses
dependances.

Pourquoi un dossier et non un fichier unique
--------------------------------------------
Le mode fichier unique existe (`--onefile`) mais decompresse 150 Mo dans un
repertoire temporaire a chaque demarrage: cinq a dix secondes d'attente avant
la premiere image, a chaque lancement. Le mode dossier demarre immediatement.
Pour une distribution en un seul fichier, il vaut mieux emballer ce dossier
dans une archive ou un installateur que de payer ce delai a chaque ouverture.

Ce que l'executable NE contient pas
-----------------------------------
La cle privee de signature, evidemment, mais aussi le module de publication
(`release.py`) et les tests: ils n'ont rien a faire chez un utilisateur.

Signature de l'executable
-------------------------
Non signe, Windows SmartScreen affichera "Windows a protege votre ordinateur"
au premier lancement, et certains antivirus mettront le fichier en
quarantaine -- les binaires PyInstaller sont un motif frequent de faux
positif. C'est une contrainte commerciale (un certificat coute de l'ordre de
100 a 400 euros par an), pas un defaut de fabrication.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "EspCamVision"

# Modules a exclure du binaire livre.
EXCLUDES = [
    "invis.release",
    "invis.test_invis",
    "matplotlib",
    "PIL",
    "pytest",
    "IPython",
    "notebook",
    "scipy",
    "pandas",
]


def entry_point() -> str:
    """Point d'entree: un fichier minuscule qui lance l'interface."""
    path = os.path.join(REPO_ROOT, "dist", "_entry.py")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            '"""Point d\'entree de l\'executable."""\n'
            "import multiprocessing, sys\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    multiprocessing.freeze_support()\n"
            "    from invis.gcs_vision import main\n"
            "    sys.exit(main())\n"
        )
    return path


# Binaires embarques par defaut mais jamais appeles par Invis. Le plus gros
# est le codec video d'OpenCV: Invis decode du JPEG depuis la memoire et
# n'ouvre aucun fichier video. Trente megaoctets pour du code mort.
#
# A ne PAS esperer d'opencv-python-headless: la variante sans interface
# graphique pese exactement autant (117,4 Mo contre 117,9 Mo), embarque le
# meme codec, et son imshow leve une erreur a l'appel. Mesure faite, gain nul.
EXCLUDE_BINARIES = (
    "opencv_videoio_ffmpeg",
)

SPEC_TEMPLATE = """# -*- mode: python ; coding: utf-8 -*-
a = Analysis([r'{entry}'], pathex=[r'{root}'], excludes={excludes!r})

_ecartes = {exclude_bin!r}
a.binaries = TOC([b for b in a.binaries
                  if not any(motif in b[0].lower() for motif in _ecartes)])

pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, exclude_binaries=True, name='{name}',
          console=False, debug=False, strip=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='{name}')
"""


def write_spec(entry: str, build_dir: str) -> str:
    os.makedirs(build_dir, exist_ok=True)
    path = os.path.join(build_dir, NAME + ".spec")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(SPEC_TEMPLATE.format(entry=entry, root=REPO_ROOT, name=NAME,
                                      excludes=list(EXCLUDES),
                                      exclude_bin=list(EXCLUDE_BINARIES)))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Construit l'executable")
    parser.add_argument("--onefile", action="store_true",
                        help="un seul fichier (demarrage nettement plus lent)")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller absent. Installe-le avec:")
        print("  python -m pip install pyinstaller")
        return 1

    build_dir = os.path.join(REPO_ROOT, "build_exe")
    dist_dir = os.path.join(REPO_ROOT, "dist")
    if args.clean:
        shutil.rmtree(build_dir, ignore_errors=True)
        shutil.rmtree(os.path.join(dist_dir, NAME), ignore_errors=True)

    spec = write_spec(entry_point(), build_dir)
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--distpath", dist_dir,
        "--workpath", build_dir,
        spec,
    ]

    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        return result.returncode

    target = os.path.join(dist_dir, NAME)
    if os.path.isdir(target):
        total = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(target) for f in fs)
        print(f"\n{target}  ({total / 1e6:.0f} Mo)")
    exe = NAME + (".exe" if sys.platform == "win32" else "")
    print(f"executable: {exe}")
    print("\nRappel: non signe -> avertissement SmartScreen au premier lancement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
