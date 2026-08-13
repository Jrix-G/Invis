# Invis

Station sol pour drone equipe d'une ESP32-CAM: distances metriques, detection
d'obstacle et reconstruction 3D de la scene, en direct, a partir d'une seule
camera.

## Demarrer

### Linux

```bash
wget https://github.com/Jrix-G/Invis/releases/latest/download/Invis-1.0.0-linux-x86_64.tar.gz
tar xzf Invis-1.0.0-linux-x86_64.tar.gz
./EspCamVision/EspCamVision
```

Le fichier a lancer s'appelle `EspCamVision`, **sans `.exe`**. S'il porte une
extension `.exe`, c'est l'archive Windows: Linux la confie a Mono, qui repond
`does not contain a valid CIL image`.

Si le programme refuse de demarrer:

```bash
sudo apt install libgl1 libglib2.0-0
```

### Windows

Telecharger
[`Invis-1.0.0-windows.zip`](https://github.com/Jrix-G/Invis/releases/latest),
decompresser, lancer `EspCamVision.exe`.

Windows affichera *« Windows a protege votre ordinateur »* au premier
lancement: cliquer sur **Informations complementaires** puis **Executer quand
meme**. L'application n'est pas signee numeriquement.

### Puis, dans l'application

1. Connecter la machine au **Wi-Fi du drone**
2. Hote `192.168.4.50`, cliquer **Connect**
3. Renseigner la **hauteur de vol** dans la barre du haut, puis *Appliquer*
4. Si l'image est a l'envers, ajuster **Miroir H** / **Miroir V**

La hauteur est la seule grandeur metrique du systeme: toutes les distances lui
sont proportionnelles. Si elle est inconnue, saisir `1.00` — les distances sont
alors exprimees en hauteurs de vol, ce qui reste exact.

Tant que l'image n'est pas droite, **les distances sont fausses**: c'est la
ligne d'image qui porte la distance.

### Sans drone

`Source` -> `sim` -> **Connect**. Un vol synthetique complet, rien a brancher.

## Depuis les sources

```bash
git clone https://github.com/Jrix-G/Invis.git
cd Invis
sudo apt install python3-tk        # Linux: Tkinter n'est pas toujours livre avec Python
pip install -r requirements.txt
python -m invis.gcs_vision
```

Les autres outils:

```bash
python -m invis.gcs_vision --source sim --connect # interface, sans drone
python -m invis.test_invis                        # 74 tests, sans materiel
python -m invis.bench --host 192.168.4.50         # mesure de cadence du lien
python -m invis.diag_stream --host 192.168.4.50   # diagnostic des coupures
python -m invis.replay <session> --height 2.4     # rejeu d'un vol enregistre
```

---

**Lecture seule cote vol** — aucun endpoint `/pilot` n'est appele, aucune
commande n'est envoyee au controleur de vol. Le programme observe, il ne
pilote pas. Le firmware de la carte vit dans un depot separe: Invis ne
dialogue avec elle qu'en HTTP.

## Les quatre panneaux

| | |
|---|---|
| **1. camera** — image, champ de vecteurs, reperes de distance au sol, ligne de contact | **2. mesures** — distances, temps avant contact, assiette, cadences, vue de dessus |
| **3. reconstruction 3D** — nuage qui se construit au fil du vol, trajectoire, drone | **4. libre** |

Sur la vue 3D: glisser pour tourner autour, molette pour zoomer. *Suivre*
recentre sur le drone, *Rotation* fait tourner la vue lentement.

Champ de vecteurs: **vert** = point qui suit le sol, **rouge** = point qui ne
le suit pas, donc du relief. La longueur des vecteurs est amplifiee, avec un
gain adapte a la vitesse pour rester lisible.

## Comment les distances sont obtenues

Une image seule ne porte aucune echelle. Ici l'echelle vient d'ailleurs: la
camera est a une hauteur connue au-dessus d'un plan. Trois mesures s'en
deduisent, dans cet ordre de disponibilite.

**1. Intersection rayon / sol.** Chaque pixel est un rayon; le sol est un
plan a la distance `h` sous la camera. `D = h / tan(alpha)`. Exact, immediat,
disponible meme en vol stationnaire.

**2. Point de contact.** Un obstacle pose au sol le touche quelque part, et ce
point-la appartient au plan: sa distance est directement metrique, sans
parallaxe, des la premiere image ou l'obstacle apparait. Attention au biais:
pres du pied, l'ecart au sol tend vers zero, donc ces points ne sont jamais
signales et le plus bas point *detecte* est toujours trop haut. Le programme
extrapole l'ecart vers zero pour retrouver la ligne de contact.

**3. Triangulation.** Pour ce qui ne touche pas le sol (branche, cable, mur vu
de face), deux visees. Le deplacement entre les deux images est mesure et mis
a l'echelle par `h`; une mediane glissante par point compense la base courte.

Precision mesuree contre la verite terrain du simulateur, sur trois
configurations (voir `test_invis.py`):

| | biais | erreur mediane |
|---|---|---|
| point de contact | −0,10 a −0,30 m | 0,14 a 0,40 m |
| triangulation | −0,02 a +0,26 m | 0,04 a 0,14 m |
| assiette retrouvee | — | 0,14 a 0,41 deg |
| odometrie sur le parcours | — | 1 a 17 % de derive |

## Si la hauteur `h` est fausse

C'est la question qui decide de tout, parce que `h` est la seule grandeur
metrique du systeme.

**L'erreur est purement multiplicative.** `D` est proportionnel a `h`: une
hauteur surestimee de 25 % donne des distances surestimees de 25 %, et rien
d'autre ne bouge. La scene n'est pas deformee, les rapports entre distances
sont exacts, l'ordre "lequel est le plus proche" est exact, le temps avant
collision est exact. Un test le verifie explicitement (`test_scale_invariance`).

Consequences pratiques:

- l'interface affiche un **intervalle** (`2.08 m [1.66 - 2.50]`) deduit de
  l'incertitude que vous declarez, plutot que des centimetres imaginaires;
- une session enregistree se **rejoue avec une autre hauteur**
  (`replay.py --height`), ce qui remet toutes les distances a l'echelle sans
  rien recalculer d'autre;
- avec `h = 1`, les distances sont exprimees en hauteurs de vol, et cette
  lecture-la est exacte quoi qu'il arrive.

**L'assiette, elle, n'est pas une simple echelle**: un degre d'erreur de
tangage deplace le point d'impact de facon non lineaire, jusqu'a 10 % de la
distance en haut d'image. Elle n'est donc pas supposee mais **mesuree sur le
sol lui-meme**, image apres image, via la normale du plan. C'est la partie de
la geometrie qu'une image *peut* determiner sans echelle. Resultat: 0,2 a
0,4 degre d'ecart typique, sans aucun capteur.

Pour passer a une echelle vraiment fiable, il faut une entree metrique:
telemetre, `RANGEFINDER` MAVLink, ou vitesse sol du FC. Toutes en lecture
seule.

## Derive et limites

- **La derive est bornee a trois axes.** Le sol donne le tangage, le roulis et
  l'altitude *sans integration*, donc sans derive. Seuls l'avance, le
  deplacement lateral et le cap s'integrent. Le cap derive (rien ne l'observe
  sans magnetometre).
- **La portee plafonne a ~2,2 fois la hauteur de vol.** A 2 m d'altitude, la
  camera piquee a 45 degres ne voit le sol que de 0,9 m a 4,4 m. Un obstacle
  se detache en pratique vers 1,4 a 1,9 m. Ce n'est pas une limite de
  l'algorithme, c'est le montage: la valeur exacte est affichee en direct
  (*portee du champ*).
- **Rien en vol stationnaire pour le relief.** Sans deplacement, pas de
  parallaxe: l'etat passe a `NO_FLOW`, la triangulation s'arrete et c'est
  affiche. Les distances au sol, elles, restent valides.
- **Aucune action.** Rien n'est transmis au FC.

## Cadence

Cout mesure sur cette machine, par image, en QVGA:

```
detecteur 4.8 ms | carte 1.3 ms | calque 1.4 ms | mesures 1.7 ms
vue 3D 1.4 ms | composition 1.6 ms | encodage 2.0 ms   => ~14 ms, plafond ~70 img/s
```

La carte en fournit environ 12 par seconde, plafond fixe par son firmware
(`max_fps=12`): le PC n'est pas le facteur limitant, et de loin.

Le capteur tourne a 5 MHz d'horloge, sous la plage habituelle de l'OV2640.
L'historique du projet indique que 10 et 20 MHz donnaient des images
corrompues y compris hors charge Wi-Fi. La cause n'est pas etablie: le
contournement de cache PSRAM a ete soupconne puis **infirme** -- le DMA PSRAM
pour la camera n'existe que sur ESP32-S2 et S3, et vaut false en dur sur
l'ESP32 classique. `bench.py` mesure, `diag_stream.py` diagnostique les
coupures; ni l'un ni l'autre ne touche a l'horloge.

Le suivi tourne en **pleine resolution QVGA**, pas en demi-resolution. Ce
n'est pas un detail de confort: pour un mur a 2,7 m survole a 2 m, l'ecart de
deplacement entre l'obstacle et le sol vaut ~3 px en 320x240 et ~1,5 px en
160x120 — sous le seuil d'ajustement du plan. Sous-echantillonner supprimait
le signal meme qu'on cherche.

## Fichiers

| Fichier | Role |
|---|---|
| `gcs_vision.py` | interface 4 panneaux, threads reseau / analyse / affichage |
| `mjpeg_client.py` | flux MJPEG decoupe par `Content-Length`, derniere image seulement, reprise |
| `detector.py` | suivi de points, deux plans ajustes, designation du sol, cellules, hysteresis |
| `geometry.py` | intrinseques, rayon/plan, assiette depuis la normale, incertitude |
| `mapper.py` | assiette, odometrie, contact au sol, triangulation, nuage de points |
| `render3d.py` | projection et rendu du nuage, en numpy |
| `panels.py` | panneau de mesures, vue de dessus, composition 2x2 |
| `overlay.py` | vecteurs, grille, reperes de distance, ligne de contact |
| `simulator.py` | monde synthetique avec verite terrain, et source video sans drone |
| `recorder.py` / `replay.py` | enregistrement CSV + images, rejeu au sol |
| `framecheck.py` | controle de qualite des images, signalement des degradations |
| `updater.py` / `release.py` | mise a jour signee, fabrication et publication |
| `build_app.py` | fabrication de l'executable autonome |
| `bench.py` / `diag_stream.py` | cadence du lien, diagnostic des coupures |
| `test_invis.py` | 74 tests, sans materiel |

## Reglages

| Reglage | Effet |
|---|---|
| `h` et `+/-` (interface) | echelle metrique et largeur de l'intervalle affiche |
| curseur *Sensi* | 0,5 (prudent) a 2,0 (nerveux) |
| `config.TTC_WARN_S` | seuil d'alerte en temps avant collision |
| `config.RESIDUAL_SIGMA_K` | severite du critere "hors sol", en ecarts-types |
| `config.CONFIRM_HITS` | severite de l'hysteresis |
| `config.CAMERA_TILT_DEG` | inclinaison nominale, point de depart de la calibration |

Regler a partir d'une session enregistree (`replay.py`), pas en vol.
