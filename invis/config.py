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
# mur ou un objet plat.
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
# Conditions exigees pour qu'une image serve a mesurer l'assiette. Bien plus
# strictes que celles qui suffisent a exploiter le plan par ailleurs, et pour
# une raison precise: un mur lointain se confond avec le sol dans
# l'homographie, mais il suffit a faire pencher la normale ajustee. Voir le
# commentaire detaille dans Mapper._update_attitude.
ATTITUDE_MIN_INLIER_RATIO = 0.95
ATTITUDE_MAX_OFF_PLANE = 0.03

# Mesure par point de contact au sol.
MIN_CONTACT_POINTS = 5
# Largeur de la bande de colonnes retenue autour de l'amas, en fraction de la
# largeur d'image. Evite de fusionner deux obstacles distincts.
CONTACT_COLUMN_BAND = 0.22
# Extrapolation de la ligne de contact, en fraction de la hauteur d'image.
# Au-dela de 1.0 le pied est sous le bord bas: l'obstacle est alors si proche
# que la mesure ne vaut plus grand-chose.
CONTACT_EXTRAPOLATION_LIMIT = 1.35

# Distance au-dela de laquelle un point reconstruit est juge non fiable.
MAX_POINT_RANGE_M = 30.0

# ---------------------------------------------------------------------------
# Structure par intersection de rayons multi-vues
# ---------------------------------------------------------------------------

# Nombre de visees avant d'accepter un point. Deux suffisent geometriquement,
# mais deux visees consecutives sont presque paralleles: la troisieme est ce
# qui fait passer d'une profondeur devinee a une profondeur mesuree.
STRUCTURE_MIN_VIEWS = 3
# Incertitude au-dela de laquelle le point est ecarte, en metres. Ce seuil
# remplace l'angle de parallaxe minimal: il porte sur la grandeur qui compte
# vraiment, l'erreur attendue en metres, et non sur un intermediaire.
STRUCTURE_MAX_SIGMA_M = 0.60
# Precision de pointage d'un detail dans l'image, en pixels. C'est elle qui
# fixe l'echelle de toutes les incertitudes calculees.
STRUCTURE_SIGMA_PX = 0.6
# Nombre de points suivis dont l'accumulation est conservee.
STRUCTURE_CAPACITY = 4096
# Fenetre d'oubli des visees, en images. Elle borne l'influence des
# observations anciennes, prises depuis une pose plus derivee et sur un point
# suivi qui a pu glisser entre-temps.
STRUCTURE_WINDOW = 12.0
# Hauteur minimale au-dessus du sol pour qu'un point compte comme obstacle.
# En dessous, c'est du relief de terrain: un caillou, une bosse, une marque de
# peinture mal ajustee par l'homographie. Un drone ne s'en preoccupe pas, et
# les inclure fausse la distance annoncee vers le bas.
OBSTACLE_MIN_HEIGHT_M = 0.15
# Le seuil monte aussi avec l'incertitude du point: un point qui ne depasse
# pas sa propre barre d'erreur n'a pas prouve qu'il n'etait pas au sol.
OBSTACLE_HEIGHT_SIGMA_K = 1.0

# Lecture de l'obstacle le plus proche.
# Duree pendant laquelle un point triangule reste pris en compte.
OBSTACLE_MEMORY_S = 2.0
# Demi-largeur du couloir devant le drone, en metres.
OBSTACLE_CORRIDOR_M = 1.5
MIN_OBSTACLE_POINTS = 6
# Incertitude au-dela de laquelle un point n'entre pas dans le calcul de la
# distance. Un point mal situe ne rend pas la mesure "un peu moins precise":
# s'il est loin devant les autres, il deplace le quantile a lui seul.
OBSTACLE_MAX_SIGMA_M = 0.35
# Quantile de distance retenu. Le minimum d'un nuage bruite est un aberrant
# par construction; un quantile bas donne la meme information sans le sursaut.
OBSTACLE_RANGE_QUANTILE = 0.15

# Nuage de points: tampon circulaire, borne pour tenir la cadence.
CLOUD_CAPACITY = 60000

# ---------------------------------------------------------------------------
# Carte d'elevation et densification du sol
# ---------------------------------------------------------------------------

# Taille de case, en metres. Dix centimetres: plus fin que cela, le bruit de
# reconstruction depasse la taille de la case et le relief affiche serait du
# bruit; plus grossier, un obstacle bas disparait dans la moyenne.
GRID_RES_M = 0.10
# Cote de la grille, en cases. 512 cases de 10 cm couvrent 51 m de cote, ce
# qui depasse largement la portee utile de cette camera. La grille suit le
# drone, donc cette valeur borne la memoire, pas la distance parcourue.
GRID_CELLS = 512
# Recentrage quand le drone depasse cette fraction du demi-cote.
GRID_RECENTRE_FRACTION = 0.25
# Poids plancher d'une mesure nouvelle. Sans plancher, une case tres observee
# cesserait d'ecouter et ne se corrigerait plus jamais -- ni apres une
# fermeture de boucle, ni si le terrain change.
GRID_MIN_WEIGHT = 0.05
# Direction de la lumiere du rendu ombre, et part de lumiere ambiante. Une
# lumiere rasante fait ressortir le relief; l'ambiante evite que les faces
# opposees deviennent illisibles.
GRID_LIGHT = (-0.55, -0.35, 0.76)
GRID_AMBIENT = 0.35
# Nombre de mesures avant qu'une case soit affichee. Une case vue une seule
# fois porte tout le bruit d'une seule mesure; en exiger deux supprime
# l'essentiel du grésillement de la surface pour une image de retard.
SURFACE_MIN_COUNT = 2
# Taille maximale d'un etalement de case, en pixels de rayon. Sans plafond,
# une case survolee de tres pres remplirait l'ecran.
SURFACE_MAX_SPLAT_PX = 6
# Hauteur correspondant au haut du degrade, en metres. Echelle fixe et non
# ajustee sur le contenu: une couleur doit toujours vouloir dire la meme
# hauteur, sinon un sol plat se peint comme un terrain accidente.
SURFACE_HEIGHT_SPAN_M = 2.0

# Densification du sol: un pixel sur N, en resolution de travail. Un point du
# sol ne demande ni parallaxe ni suivi -- son rayon perce un plan connu -- donc
# rien n'oblige a se limiter aux points suivis. Le pas fixe le compromis entre
# densite de la surface et cout: 6 donne environ 2000 points par image pour le
# prix d'une seule operation vectorisee.
DENSE_ENABLED = True
DENSE_STEP_PX = 6
# Rayon d'exclusion autour d'un point signale hors sol, en pixels de travail.
# La ou il y a du relief, l'intersection avec le plan poserait l'obstacle a
# plat et derriere lui.
DENSE_OBSTACLE_MARGIN_PX = 8
# Portee au-dela de laquelle un point dense n'est pas verse: pres de
# l'horizon, une erreur d'assiette d'un degre deplace le point de plusieurs
# metres, et la surface se couvrirait d'un voile faux.
DENSE_MAX_RANGE_M = 8.0
# Marge sous l'horizon, en fraction de la hauteur d'image, ou l'on ne verse
# rien: c'est la zone ou la geometrie devient indeterminee.
DENSE_HORIZON_MARGIN = 0.10
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
# Filtrage de la pose
# ---------------------------------------------------------------------------

# Le filtre remplace l'integration directe du deplacement mesure. Desactivable
# pour comparer, mais il n'y a pas de raison de s'en priver: il coute quelques
# dizaines d'operations par image et supprime l'essentiel du bruit d'odometrie.
KALMAN_ENABLED = True
# Acceleration possible du drone, en m/s^2. C'est ce qui dit au filtre a quel
# point la vitesse peut changer entre deux images: trop bas, il ignore les
# vraies accelerations; trop haut, il ne filtre plus rien.
KALMAN_ACCEL_SIGMA = 1.5
# Bruit de la mesure de vitesse issue de l'homographie, en m/s.
KALMAN_VELOCITY_SIGMA = 0.30
# Seuil de rejet, en ecarts-types. Au-dela, la mesure est jugee incompatible
# avec l'etat et ecartee: c'est le vrai garde-fou contre une homographie
# ajustee sur un mur.
KALMAN_GATE_SIGMA = 3.5
# Meme reglage pour le cap, en degres par seconde.
KALMAN_YAW_ACCEL_DPS2 = 60.0
KALMAN_YAW_SIGMA_DPS = 10.0
# Vitesse nulle observee. Une scene immobile n'est pas une absence de mesure:
# c'est la mesure "je ne bouge pas", et c'est l'une des plus fiables du
# systeme. L'exploiter empeche la vitesse filtree de continuer sur sa lancee
# pendant un stationnaire.
KALMAN_ZUPT_SIGMA = 0.10
KALMAN_ZUPT_YAW_DPS = 3.0

# ---------------------------------------------------------------------------
# Entree inertielle (facultative)
# ---------------------------------------------------------------------------

# Le systeme fonctionne sans centrale inertielle: c'est sa raison d'etre. Mais
# si le controleur de vol en publie une, deux grandeurs valent d'etre prises.
#
# Bruit du gyrometre de lacet, en degres par seconde. Celui d'un MEMS de
# petit multirotor. Compare aux ~10 deg/s de la mesure visuelle, il domine
# largement -- c'est ce qui fait qu'il freine la derive de cap.
IMU_GYRO_SIGMA_DPS = 1.5
# Age au-dela duquel une mesure inertielle n'est plus utilisee, en secondes.
# Une assiette vieille d'une demi-seconde ne decrit plus le drone: mieux vaut
# revenir a l'estimation visuelle que de recaler sur du passe.
IMU_MAX_AGE_S = 0.25

# ---------------------------------------------------------------------------
# Fermeture de boucle
# ---------------------------------------------------------------------------

LOOP_ENABLED = True
# Vignette de sol redressee: taille en pixels et portee metrique. 8 m pour
# 64 px font 12,5 cm par pixel -- la precision de recalage qu'on peut esperer.
# Plus fin ne servirait a rien: le sol n'est pas connu a mieux que cela.
LOOP_PATCH_PX = 64
# 6 m pour 64 px font 9,4 cm par pixel. La portee fixe aussi le decalage
# maximal rattrapable -- la correlation de phase repond jusqu'a environ la
# moitie de la vignette, soit trois metres de derive.
LOOP_PATCH_SPAN_M = 6.0
# Descripteur: la vignette reduite. 16x16 suffit a distinguer des lieux, et
# rend la comparaison a la memoire entiere plus rapide que le redressement.
LOOP_DESCRIPTOR_PX = 16
# Fraction de la vignette qui doit etre reellement observee. En dessous, la
# vue de dessus est surtout du vide et se ressemble partout.
#
# Le plafond physique est bas et il faut le savoir: la vignette est centree
# sur le drone alors que la camera ne voit qu'un trapeze devant lui. La moitie
# arriere est structurellement vide, et le champ de 54 degres ne remplit pas
# l'autre moitie. Mesure entre 11 % et 17 % selon l'altitude, jamais plus. Le
# seuil est donc cale sous ce plafond, pas sur une intuition de "moitie
# remplie".
LOOP_MIN_COVERAGE = 0.08
# Espacement des lieux memorises, en metres. La memoire couvre une distance,
# pas une duree: un stationnaire n'ajoute rien.
LOOP_KEY_SPACING_M = 0.6
LOOP_CAPACITY = 256
# Anciennete minimale d'un lieu pour qu'il puisse etre "reconnu". Plus recent,
# il confirmerait l'odometrie par elle-meme au lieu de la corriger.
LOOP_MIN_AGE_S = 8.0
# Rayon de recherche de base, elargi ensuite par l'incertitude accumulee.
LOOP_SEARCH_RADIUS_M = 2.0
# Ressemblance minimale des descripteurs. Volontairement exigeante: une
# fausse fermeture est bien plus nuisible qu'une fermeture manquee.
LOOP_MIN_SIMILARITY = 0.55
# Nettete du pic de correlation de phase. Un pic mou signifie que les deux
# vignettes ne decrivent pas le meme sol, quoi qu'en dise le descripteur --
# ou que le cap estime est trop faux pour qu'elles soient comparables. Sur des
# decalages verifies allant jusqu'a trois metres, un vrai recouvrement donne
# au moins 0,36; on garde de la marge pour les images reelles, plus bruitees.
LOOP_MIN_PHASE_RESPONSE = 0.15
# Incertitude attribuee a la position issue d'une fermeture. De l'ordre du
# pixel de vignette: c'est la resolution du recalage, pas mieux.
LOOP_POSITION_SIGMA_M = 0.20

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
