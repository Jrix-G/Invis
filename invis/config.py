"""Configuration statique du client vision.

Toutes les valeurs qui decrivent le drone, l'optique ou les seuils de
detection sont regroupees ici. Aucun module ne doit coder ces nombres en dur.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reseau
# ---------------------------------------------------------------------------

# Adresse du pont ESP32-CAM sur le reseau local du drone.
DEFAULT_HOST = "192.168.4.50"
DEFAULT_PORT = 80

STREAM_PATH = "/stream"
SNAPSHOT_PATH = "/jpg"
STATUS_PATH = "/status"
CONTROL_PATH = "/control"

# Le firmware n'accepte qu'un seul client video (VIDEO_MAX_CLIENTS = 1) et
# prend la place de l'ancien flux au bout de VIDEO_PREEMPT_TIMEOUT_MS. Se
# connecter ici coupe donc l'onglet navigateur eventuellement ouvert.
CONNECT_TIMEOUT_S = 4.0
READ_TIMEOUT_S = 6.0
# Attente avant reprise du flux. La carte coupe la connexion toutes les
# quelques secondes (IncompleteRead cote client): une coupure est donc un
# evenement courant, pas une panne. La premiere reprise doit etre quasi
# immediate, sinon le temps mort depasse le temps de flux utile.
RECONNECT_BACKOFF_S = (0.05, 0.2, 0.6, 1.5, 3.0)
# Duree au-dela de laquelle un flux est juge "a fonctionne", ce qui remet le
# decompte des tentatives a zero.
RECONNECT_STABLE_S = 1.5

# Taille maximale d'une image JPEG acceptee. Au-dela on considere le tampon
# desynchronise et on repart d'un marqueur SOI propre.
MAX_JPEG_BYTES = 512 * 1024

# ---------------------------------------------------------------------------
# Optique / geometrie
# ---------------------------------------------------------------------------

# OV2640 avec l'objectif standard des cartes AI-Thinker.
HFOV_DEG = 54.0
VFOV_DEG = 41.0

# Inclinaison de l'axe optique par rapport a l'horizontale, en degres.
# Negatif = camera piquee vers le sol (cas nominal du drone).
CAMERA_TILT_DEG = -45.0

# Orientation du capteur. La camera de ce drone rend l'image retournee; il
# faut la redresser AVANT toute analyse, car la ligne d'image porte la
# distance et un miroir simple inverse le sens du repere. Reglable en direct
# depuis l'interface (cases "Miroir H" et "Miroir V").
CAMERA_FLIP_H = True
CAMERA_FLIP_V = True

# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------

# Resolution de travail: la pleine resolution du flux.
#
# Diviser par deux paraissait gratuit -- ca ne l'est pas. Ce qu'on cherche,
# c'est l'ecart de deplacement entre un obstacle et le sol derriere lui. Pour
# un mur a 2,7 m survole a 2 m de haut, cet ecart vaut environ 3 px en 320x240
# et donc 1,5 px en 160x120: sous le seuil d'ajustement du plan, l'obstacle
# devenait indiscernable du sol. Le sous-echantillonnage supprimait le signal
# meme qu'on veut mesurer. En QVGA le suivi coute quelques millisecondes: la
# depense est negligeable devant la detection perdue.
WORK_WIDTH = 320

# Grille d'analyse (colonnes x lignes).
GRID_COLS = 3
GRID_ROWS = 3

# Suivi Lucas-Kanade.
MAX_FEATURES = 200
MIN_FEATURES = 40
FEATURE_QUALITY = 0.02
FEATURE_MIN_DISTANCE = 5
# Recherche de nouveaux points toutes les N images. Sans cela, un objet qui
# entre dans le champ n'est jamais echantillonne: on ne suivrait que les points
# nes a la premiere image, et le detecteur resterait aveugle a ce qui arrive.
REDETECT_EVERY = 3
LK_WINDOW = 15
LK_LEVELS = 3
# Tolerance du controle aller-retour, en pixels a la resolution de travail.
FB_ERROR_MAX = 1.0

# Tolerance d'ajustement du plan dominant par RANSAC (pixels).
# Serree volontairement: elle fixe la plus petite difference de profondeur
# encore visible, donc la distance a laquelle un obstacle se detache du sol.
HOMOGRAPHY_RANSAC_PX = 1.2

# Seuil de decision "hors plan". Il ne peut pas etre fixe: a 5-8 images par
# seconde le deplacement apparent ne fait qu'environ un pixel, un seuil fixe de
# 2 pixels ne se declencherait jamais.
#
# Il est cale sur la dispersion reelle des residus du plan, pas sur
# l'amplitude du flux: ce qu'on veut savoir, c'est "ce point s'ecarte-t-il du
# sol plus que le bruit d'ajustement", question qui n'a rien a voir avec la
# vitesse d'avance. Un seuil proportionnel au flux devenait au contraire de
# plus en plus permissif a mesure qu'on accelerait, et laissait passer les
# obstacles lointains.
RESIDUAL_MIN_PX = 0.8
RESIDUAL_SIGMA_K = 4.0
# Repli quand la dispersion n'est pas mesurable (trop peu de points).
RESIDUAL_FLOW_RATIO = 0.5

# Fraction minimale de points compatibles avec le plan pour croire que le plan
# trouve est bien le sol. En dessous, l'homographie a probablement accroche un
# mur ou un objet plat: on ne s'en sert pas pour recalibrer l'assiette.
PLANE_INLIER_MIN_RATIO = 0.5
# Nombre minimal de points pour tenter un modele plan.
MIN_POINTS_FOR_MODEL = 12
# Nombre minimal de points dans une cellule pour la juger.
MIN_POINTS_PER_CELL = 5

# Fraction de points hors plan dans une cellule au-dela de laquelle la cellule
# est suspecte.
CELL_OUTLIER_RATIO = 0.45

# Temps avant collision (secondes) en dessous duquel on alerte.
TTC_WARN_S = 2.5
# Expansion minimale exploitable: sous ce seuil le rapport d'echelle est du
# bruit, pas un rapprochement.
MIN_SCALE_RATE = 0.004

# Sous ce deplacement median (pixels/frame), la scene est consideree immobile:
# le flux optique ne porte aucune information de profondeur.
STATIC_FLOW_PX = 0.35

# Hysteresis: il faut CONFIRM_HITS detections dans les CONFIRM_WINDOW dernieres
# images pour declarer un obstacle, et autant d'images vides pour le lever.
CONFIRM_WINDOW = 6
CONFIRM_HITS = 4
RELEASE_MISSES = 6

# ---------------------------------------------------------------------------
# Sortie
# ---------------------------------------------------------------------------

# Repertoire des sessions enregistrees (CSV + images brutes).
SESSION_DIR = "sessions"

# ---------------------------------------------------------------------------
# Telemetrie sol (P5) et reconstruction 3D (P6)
# ---------------------------------------------------------------------------

# Hauteur de vol par defaut, en metres, modifiable dans l'interface.
DEFAULT_HEIGHT_M = 2.0
# Incertitude assumee sur cette hauteur. Le barometre d'un petit multirotor
# derive facilement de cet ordre; l'afficher evite d'annoncer une precision
# qui n'existe pas.
DEFAULT_SIGMA_H_M = 0.4

# Pente minimale d'un rayon vers le bas pour accepter une intersection sol.
# En dessous, le point d'impact part vers l'horizon et devient ininterpretable.
MIN_RAY_SLOPE = 0.06

# Calibration d'assiette a partir de la normale du plan mesuree.
#
# L'estimation image par image est bruitee et, quand un obstacle grandit dans
# le champ, franchement fausse par moments. Un lissage exponentiel suit ces
# excursions; une mediane glissante les ignore tant qu'elles restent
# minoritaires. L'inclinaison est mecanique, elle ne saute pas: la mediane est
# le bon outil.
ATTITUDE_WINDOW = 15
ATTITUDE_MIN_SAMPLES = 5
ATTITUDE_SMOOTHING = 0.25
# Ecart maximal accepte par rapport a l'inclinaison nominale, en degres.
# Au-dela on considere que l'homographie n'a pas trouve le sol.
ATTITUDE_MAX_DEVIATION_DEG = 25.0

# Mesure par point de contact au sol.
MIN_CONTACT_POINTS = 5
# Largeur de la bande de colonnes retenue autour de l'amas, en fraction de la
# largeur d'image. Evite de fusionner deux obstacles distincts.
CONTACT_COLUMN_BAND = 0.22
# Extrapolation de la ligne de contact, en fraction de la hauteur d'image.
# Au-dela de 1.0 le pied est sous le bord bas: l'obstacle est alors si proche
# que la mesure ne vaut plus grand-chose.
CONTACT_EXTRAPOLATION_LIMIT = 1.35

# Odometrie visuelle / triangulation.
# Angle minimal entre les deux rayons, en degres. Sous cet angle la
# triangulation est mal conditionnee et produit des points aberrants.
MIN_PARALLAX_DEG = 1.2
# Distance au-dela de laquelle un point triangule est juge non fiable.
MAX_POINT_RANGE_M = 30.0

# Lecture de l'obstacle le plus proche.
# Duree pendant laquelle un point triangule reste pris en compte.
OBSTACLE_MEMORY_S = 2.0
# Demi-largeur du couloir devant le drone, en metres.
OBSTACLE_CORRIDOR_M = 1.5
MIN_OBSTACLE_POINTS = 6
# Quantile de distance retenu. Le minimum d'un nuage bruite est un aberrant
# par construction; un quantile bas donne la meme information sans le sursaut.
OBSTACLE_RANGE_QUANTILE = 0.15

# Nuage de points: tampon circulaire, borne pour tenir la cadence.
CLOUD_CAPACITY = 60000
# Nombre de poses conservees pour la trajectoire affichee.
TRAJECTORY_CAPACITY = 4000

# Rendu 3D.
VIEW3D_SIZE = (480, 360)
VIEW3D_DEFAULT_YAW_DEG = -35.0
VIEW3D_DEFAULT_PITCH_DEG = 28.0
VIEW3D_DEFAULT_RANGE_M = 8.0

CELL_NAMES = [
    ["HG", "HC", "HD"],
    ["MG", "MC", "MD"],
    ["BG", "BC", "BD"],
]

# Mediane glissante des profondeurs par point suivi. Une base courte donne une
# profondeur bruitee; repeter la mesure sur quelques images et en prendre la
# mediane recupere l'essentiel de la precision perdue.
DEPTH_HISTORY = 7
DEPTH_MIN_SAMPLES = 3

# Ressemblance minimale entre la normale trouvee et celle attendue pour le sol
# (produit scalaire). En dessous, le plan ajuste n'est pas le sol et la pose
# qui en decoule n'a pas de sens.
GROUND_NORMAL_MIN_SCORE = 0.5

# Vue de dessus du panneau de mesures: portee affichee, en metres.
RADAR_SPAN_M = 5.0
# Distances marquees sur l'image camera, en metres.
RANGE_TICKS_M = (1.0, 2.0, 3.0, 4.0, 6.0)

# Longueur visee des vecteurs de flux a l'affichage, en pixels.
FLOW_TARGET_PX = 14.0
# Vitesse au-dela de laquelle l'estimation est jugee fausse, en m/s.
MAX_SPEED_MPS = 3.0

# ---------------------------------------------------------------------------
# Controle de qualite des images
# ---------------------------------------------------------------------------

# Resolution d'analyse du controle qualite. Les defauts recherches sont
# grossiers: inutile de les chercher en pleine resolution.
QUALITY_WIDTH = 160
# Nombre d'images servant de reference glissante.
QUALITY_WINDOW = 20

# Nettete, en fraction de la nettete habituelle du flux. Le critere est
# relatif: la nettete "normale" depend de la scene, de la lumiere et de la
# qualite JPEG du moment, elle ne se fixe pas dans l'absolu.
QUALITY_SHARPNESS_WARN = 0.55
QUALITY_SHARPNESS_REJECT = 0.32
# Taille JPEG minimale par rapport a l'habitude: en dessous, image tronquee.
QUALITY_SIZE_REJECT = 0.35
# Ecart d'un canal a la luminance au-dela duquel un pixel est juge aberrant.
QUALITY_CHROMA_PIXEL = 70
# Exces de pixels aberrants par rapport a l'habitude du flux, en fraction de
# l'image. Relatif et non absolu: une scene naturellement coloree depasse en
# permanence un seuil absolu, alors qu'une corruption se reconnait a ce
# qu'elle surgit. Les valeurs sont basses parce que quelques pour cent
# d'image corrompue suffisent a fausser l'ajustement du plan.
QUALITY_CHROMA_WARN = 0.008
QUALITY_CHROMA_REJECT = 0.020

# Nombre de rejets consecutifs au-dela duquel on considere que la scene est
# reellement pauvre en details, et non que les images sont ratees.
QUALITY_MAX_STREAK = 8
# Nombre maximal d'images ecartees conservees sur disque, pour inspection.
QUALITY_DUMP_MAX = 60
