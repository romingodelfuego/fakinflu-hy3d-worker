#!/usr/bin/env python3
"""fakinflu — worker RunPod Hunyuan3D-2 (texture / forme HD + texture).

Entree : {"items": [ {id, mode, image, mesh?, seed?, steps?, octree?, max_faces?, skip_rembg?}, ... ]}
         (un item seul a plat dans "input" est aussi accepte)
Sortie : {"results": [ {id, glb: <b64>, stats: {...}} | {id, error: "..."} ]}

Plusieurs items par job = un seul cold start paye. Les poids sont BAKES dans
l'image (HY3DGEN_MODELS) : aucun telechargement au runtime.
"""
import base64, io, os, time, traceback

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import runpod
import torch
import trimesh
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline

MODEL = os.environ.get("HY3D_MODEL", "tencent/Hunyuan3D-2")
PAINT_SUBFOLDER = os.environ.get("HY3D_PAINT_SUBFOLDER", "hunyuan3d-paint-v2-0-turbo")
SHAPE_SUBFOLDER = os.environ.get("HY3D_SHAPE_SUBFOLDER", "hunyuan3d-dit-v2-0-turbo")
JPEG_QUALITY = int(os.environ.get("TEXTURE_JPEG_QUALITY", "90"))

# --- chargement unique au cold start ------------------------------------------
# Paint sert dans les deux modes -> charge tout de suite.
# La forme HD n'est chargee qu'au premier job "full" puis gardee en memoire.
_paint = Hunyuan3DPaintPipeline.from_pretrained(MODEL, subfolder=PAINT_SUBFOLDER)
_rembg = BackgroundRemover()
_shape = None


def _shape_pipe():
    global _shape
    if _shape is None:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        _shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL, subfolder=SHAPE_SUBFOLDER, use_safetensors=True, device="cuda")
        _shape.enable_flashvdm(mc_algo="mc")
    return _shape


def _b64decode(b64: str) -> bytes:
    if b64.strip().startswith("data:") and "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def _jpeg_textures(mesh):
    """Re-encode la texture en JPEG : trimesh garde le JPEG tel quel a l'export
    GLB (au lieu d'un PNG 3-5x plus lourd) -> sortie du job bien plus legere."""
    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return
    for attr in ("baseColorTexture", "image"):
        img = getattr(mat, attr, None)
        if isinstance(img, Image.Image):
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
            buf.seek(0)
            setattr(mat, attr, Image.open(buf))


def _process(item: dict) -> dict:
    stats = {}
    mode = item.get("mode", "texture")

    image = Image.open(io.BytesIO(_b64decode(item["image"])))
    if not item.get("skip_rembg"):
        image = _rembg(image.convert("RGB"))

    if mode == "texture":
        if not item.get("mesh"):
            return {"error": "mode texture : champ 'mesh' (glb base64) requis"}
        mesh = trimesh.load(io.BytesIO(_b64decode(item["mesh"])), file_type="glb", force="mesh")
    elif mode == "full":
        t = time.time()
        mesh = _shape_pipe()(
            image=image,
            num_inference_steps=int(item.get("steps", 5)),
            octree_resolution=int(item.get("octree", 380)),
            num_chunks=20000,
            generator=torch.Generator().manual_seed(int(item.get("seed", 0))),
            output_type="trimesh",
        )[0]
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        max_faces = int(item.get("max_faces", 60000))
        if len(mesh.faces) > max_faces:
            mesh = FaceReducer()(mesh, max_facenum=max_faces)
        stats["shape_s"] = round(time.time() - t, 1)
    else:
        return {"error": f"mode inconnu : {mode} (texture|full)"}

    stats["faces"] = len(mesh.faces)
    t = time.time()
    mesh = _paint(mesh, image=image)
    stats["texture_s"] = round(time.time() - t, 1)

    _jpeg_textures(mesh)
    glb = mesh.export(file_type="glb")
    stats["glb_bytes"] = len(glb)
    return {"glb": base64.b64encode(glb).decode(), "stats": stats}


def handler(job):
    inp = job.get("input", {}) or {}
    items = inp.get("items") or ([inp] if inp.get("image") else [])
    if not items:
        return {"error": "champ 'items' (liste) ou 'image' requis"}

    results = []
    for item in items:
        res = {"id": item.get("id")}
        if not item.get("image"):
            res["error"] = "champ 'image' (base64) requis"
        else:
            try:
                res.update(_process(item))
            except Exception as e:  # un item rate ne fait pas echouer tout le lot
                traceback.print_exc()
                res["error"] = f"{type(e).__name__}: {e}"
            finally:
                torch.cuda.empty_cache()
        results.append(res)
    return {"results": results}


runpod.serverless.start({"handler": handler})
