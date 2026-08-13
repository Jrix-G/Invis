# Distribution et mises a jour

L'application se distribue en executable autonome, et se met a jour par
archives signees que l'utilisateur accepte d'un clic.

## Pourquoi deux choses separees

```
executable   ~179 Mo   Python + OpenCV + numpy   change rarement
payload        66 Ko   le code du projet          change souvent
```

Un rapport de 1 a 2700. Mettre a jour l'executable entier ferait telecharger
179 Mo pour corriger trois lignes. Seul le payload circule.

## Preparer la publication, une fois pour toutes

```bash
python -m invis.release keygen
```

Ecrit la cle privee **hors du depot** et affiche la cle publique. Colle cette
derniere dans `updater.py` :

```python
PUBLIC_KEY_HEX = "a35685ead2....."
MANIFEST_URL = "https://github.com/TON_COMPTE/TON_DEPOT/releases/latest/download/manifest.json"
```

Deux regles sur la cle privee : elle ne rentre jamais dans le depot (`*.pem`
est ignore), et elle se sauvegarde. **La perdre coupe definitivement toute
possibilite de publier une mise a jour** aux utilisateurs deja installes : leur
programme refusera toute archive signee par une autre cle. C'est le
comportement voulu, mais il est sans recours.

## Construire l'executable

```bash
python -m invis.build_app --clean
```

Produit `dist/EspCamVision/`. Fonctionne sur Windows et Linux, mais **il faut
construire sur chaque systeme** : PyInstaller n'effectue pas de compilation
croisee. Un binaire Windows se fabrique sous Windows, un binaire Linux sous
Linux.

Mode dossier par defaut, demarrage immediat. `--onefile` donne un fichier
unique mais decompresse 179 Mo a chaque lancement, soit cinq a dix secondes
d'attente. Pour distribuer un seul fichier, emballe plutot le dossier dans une
archive ou un installateur.

### L'avertissement Windows

Sans certificat de signature de code, le premier lancement affiche *« Windows
a protege votre ordinateur »* : l'utilisateur doit cliquer sur *Informations
complementaires* puis *Executer quand meme*. Certains antivirus peuvent aussi
placer le fichier en quarantaine, les binaires PyInstaller etant un motif
frequent de faux positif.

C'est une contrainte commerciale, pas un defaut de fabrication : la lever
demande un certificat, de l'ordre de 100 a 400 euros par an. A prevoir dans la
notice tant que ce n'est pas le cas.

## Publier une mise a jour

1. Incremente `VERSION` dans `invis/version.py`.
2. Fabrique et signe :

```bash
python -m invis.release package --version 1.1.0
python -m invis.release sign --version 1.1.0 \
    --key ../esp32cam-vision-signing.pem \
    --base-url https://github.com/TON_COMPTE/TON_DEPOT/releases/download/v1.1.0 \
    --notes "Correction du decoupage du flux video"
```

3. Depose `dist/payload-1.1.0.zip` et `dist/manifest.json` sur la release.

L'application consulte le manifeste au demarrage, affiche un bandeau
*« Version 1.1.0 disponible — Installer / Plus tard »*, et n'installe que sur
clic.

## Ce que le programme refuse d'installer

Une mise a jour automatique est de l'execution de code a distance : qui
controle l'adresse de publication controle les machines qui la consultent. Et
cette application dialogue avec un drone.

Chaque refus ci-dessous est verifie par un test (`test_updater`) :

| Situation | Comportement |
|---|---|
| Archive modifiee apres signature | refusee |
| Signature d'une autre cle | refusee |
| Empreinte SHA-256 incoherente | refusee |
| Version identique ou anterieure | refusee |
| Archive ou manifeste servis en HTTP | refuses |
| Aucune cle publique compilee | refusee |
| Archive au chemin remontant (`../..`) | refusee |

Une empreinte seule ne suffirait pas : qui remplace l'archive sur le serveur
remplace aussi son empreinte. Seule une signature qu'il ne peut pas fabriquer
l'arrete. L'empreinte reste verifiee, mais contre la corruption de transfert,
pas contre une attaque.

Rien n'est ecrit a l'emplacement definitif avant validation de la signature, et
l'installation se fait par renommage : soit la nouvelle version est
entierement en place, soit l'ancienne reste intacte. Les deux dernieres
versions sont conservees.

## Ou vont les fichiers

| Systeme | Emplacement |
|---|---|
| Windows | `%LOCALAPPDATA%\esp32cam-vision\payload\<version>\` |
| Linux | `~/.local/share/esp32cam-vision/payload/<version>/` |

Espace utilisateur, jamais le dossier d'installation : une mise a jour ne
demande donc aucun droit administrateur, et fonctionne meme si le programme
est installe en lecture seule.

## Le bandeau n'apparait jamais en cours de vol

Tant que l'application est connectee a la camera, la proposition reste
masquee. Changer le logiciel pendant qu'on s'en sert est le meilleur moyen de
transformer une mise a jour en panne inexpliquee — et de faire chercher une
panne materielle pendant deux heures.
