import subprocess
from pathlib import Path

def export_blend_to_glb(blend_path: Path, glb_path: Path, blender_bin: Path) -> None:
    """Export a .blend scene to GLB using Blender in headless mode."""
    if glb_path.exists():
        return
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(blender_bin),
        "-b",
        str(blend_path),
        "--python-expr",
        (
            "import bpy; "
            "bpy.ops.export_scene.gltf(filepath=r'%s', export_format='GLB')"
        )
        % glb_path,
    ]
    subprocess.run(cmd, check=True)