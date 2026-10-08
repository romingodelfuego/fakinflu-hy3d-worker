# Déploiement de l'endpoint hy3d (Hunyuan3D-2, RunPod Serverless, SANS volume)

Objectif : un endpoint qui **texture** les formes 3D itérées en local sur le Mac
(`make shape`), ou qui refait une **forme HD** puis la texture. Le GLB texturé revient
**dans la réponse du job** et `hy3d/hy3d_runpod.py` l'écrit dans `data/out/3d/<id>/`.

| Mode | Ce que fait le GPU | Modèles |
|---|---|---|
| `texture` (`make texture ID=…`) | texture du mesh envoyé (déjà nettoyé et décimé en local) | Delight v2-0 + Paint v2-0-Turbo |
| `full` (`make shape-hd ID=…`) | forme HD (même image détourée, même seed) puis texture | DiT v2-0-Turbo + VAE turbo (FlashVDM) + Paint |

## Ce qui rend chaque requête la moins chère possible

- **Préparé en local, gratuitement** : détourage (rembg), nettoyage et décimation du mesh
  (`--max-faces`). Le worker reçoit un PNG détouré plus un GLB de quelques centaines de Ko.
- **Batch** : `make texture ID="a b c"` envoie **une seule requête**. Le cold start (~1-2 min)
  n'est payé qu'une fois et le handler traite les items à la suite. Si un item échoue, les
  autres continuent.
- **Poids bakés dans l'image** (~17 Go) : pas de volume réseau (0 $ de stockage), pas de
  téléchargement au cold start, et FlashBoot efficace. On ne bake que les sous-dossiers
  utiles, en safetensors seulement : tout `tencent/Hunyuan3D-2` ferait plus de 60 Go.
- **Forme HD chargée à la demande** : un worker qui ne fait que de la texture ne charge pas DiT.
- **Texture en JPEG** dans le GLB : sortie 3 à 5 fois plus légère qu'en PNG.
- **Garde-fou côté client** : un payload de plus de 9 Mo est refusé **avant** l'envoi.

## Réglages de l'endpoint (coût-optimaux, comme les autres)

- GPU **24 Go** (pool Ampere 24 / L4 / A5000, environ 0,69 $/h). Paint, Delight et DiT
  tiennent ensemble dans moins de 20 Go.
- **min workers 0**, **max workers 1–2**, **idle timeout ~5 s**, **FlashBoot on**.
- **Execution timeout 600 s**. Un batch de N meshes prend environ N × 1 min.
- Container disk ≥ 40 Go.
- Ordre de grandeur : texture ≈ 1 min/mesh à chaud, cold start inclus ≈ 0,02–0,04 $
  par requête d'un mesh.

## Build (poids bakés → build côté RunPod)

Comme pour promptgen (CONTRIBUTING §6) : **repo GitHub dédié**, lié à RunPod, et rebuild à
chaque **GitHub Release**.

1. Pousser `hy3d/serverless/` à plat dans un repo (par exemple `fakinflu-hy3d-worker`).
2. Console RunPod → **Serverless → New Endpoint → Import Git Repository** → ce repo.
   - Build args optionnels : `HY3D_REF=<commit Hunyuan3D-2>` (épinglé par défaut) et `CUDA_TAG=…`.
   - Le build compile `custom_rasterizer` et `differentiable_renderer` (CUDA, ~5 min), puis
     télécharge ~17 Go de poids. On reste dans les limites RunPod (30 min de `docker build`,
     80 Go d'image).
3. Appliquer les réglages ci-dessus, puis récupérer l'**Endpoint ID** →
   `fakinflu/.env` (`RUNPOD_HY3D_ENDPOINT_ID`).
4. `make hy3d-check` (0 crédit) → `make texture ID=<id>`.

**Variante : build local + push** (machine x86 avec Docker, pas le Mac M1) :
```bash
cd hy3d/serverless
docker build --platform linux/amd64 -t <toncompte>/fakinflu-hy3d:v1 .
docker push <toncompte>/fakinflu-hy3d:v1
```

## Contrat d'API

Entrée (`input`) :
```json
{"items": [
  {"id": "20261004_101500_ab12", "mode": "texture", "image": "<b64 png détouré>", "mesh": "<b64 glb>", "skip_rembg": true},
  {"id": "20261004_101900_cd34", "mode": "full", "image": "<b64>", "seed": 1234, "steps": 5, "octree": 380, "max_faces": 60000, "skip_rembg": true}
]}
```
Sortie :
```json
{"results": [
  {"id": "…", "glb": "<b64 glb texturé>", "stats": {"faces": 40000, "texture_s": 48.2, "glb_bytes": 3100000}},
  {"id": "…", "error": "RuntimeError: …"}
]}
```

Variables d'environnement du worker :
- `HY3D_PAINT_SUBFOLDER` (défaut `hunyuan3d-paint-v2-0-turbo`) ;
- `HY3D_SHAPE_SUBFOLDER` (défaut `hunyuan3d-dit-v2-0-turbo`) ;
- `TEXTURE_JPEG_QUALITY` (défaut 90).

> ⚠️ **Licence** : Hunyuan3D-2 est sous *Tencent Hunyuan 3D 2.0 Community License*.
> Le « Territory » **exclut l'Union européenne, le Royaume-Uni et la Corée du Sud** : aucun
> droit d'usage n'y est accordé, ni pour les poids ni pour les sorties (§1.l, §5.c, AUP §1).
> Au-delà d'un million d'utilisateurs actifs mensuels (MAU), une licence Tencent dédiée est
> nécessaire. À valider avant tout usage depuis la France.
