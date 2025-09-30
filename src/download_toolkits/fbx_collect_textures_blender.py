import bpy, sys, os, json, shutil, re

def _clean_path_for_manifest(fp: str) -> str:
    """把 Windows 盘符/UNC/越级路径清洗为可读相对形式（优先 basename）。"""
    s = (fp or "").strip().replace("\\","/")
    if not s:
        return s
    # 盘符或 UNC：仅保留文件名
    if re.match(r"^[A-Za-z]:/", s) or s.startswith("//") or s.startswith("\\\\"):
        return os.path.basename(s)
    # Blender 在 *nix 上可能把 C:/... 当相对片段拼到前缀目录后面，这里剥离掉
    if "/C:/" in s or s.startswith("C:/") or s.startswith("C:"):
        return os.path.basename(s)
    # 越级也只留文件名
    if "/../" in f"/{s}":
        return os.path.basename(s)
    # 至多保留最后两段，避免把很长的上游路径写进清单
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return s

def main():
    argv = sys.argv
    if "--" not in argv or len(argv[argv.index("--")+1:]) < 2:
        print("USAGE: blender --background --python fbx_collect_textures_blender.py -- <fbx_path> <out_dir>")
        return
    fbx_path, out_dir = argv[argv.index("--")+1:][0], argv[argv.index("--")+1:][1]
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=os.path.abspath(fbx_path))

    exported = []
    referenced = []

    for img in list(bpy.data.images):
        try:
            fp = img.filepath if img.filepath else ""
            try:
                abspath = bpy.path.abspath(fp) if fp else ""
            except Exception:
                abspath = fp
            # 名称一律用解析后的绝对路径 basename；没有扩展就补 .png
            name = os.path.basename(abspath) if abspath else (img.name or "Image")
            if not os.path.splitext(name)[1]:
                name += ".png"

            # 清洗后的展示路径 + 保留原始原样路径（便于排查）
            clean_fp = _clean_path_for_manifest(fp)
            referenced.append({"name": name, "filepath": clean_fp, "orig": fp})

            dst = os.path.join(out_dir, os.path.basename(name))
            if abspath and os.path.exists(abspath):
                if not os.path.exists(dst):
                    shutil.copy2(abspath, dst)
                exported.append(os.path.basename(dst))
            else:
                try:
                    base, ext = os.path.splitext(name)
                    if not ext or ext.lower() not in [".png",".jpg",".jpeg",".tga",".bmp",".tiff",".exr",".dds",".webp"]:
                        name = base + ".png"
                        dst = os.path.join(out_dir, name)
                        img.file_format = "PNG"
                    else:
                        dst = os.path.join(out_dir, name)
                    img.filepath_raw = dst
                    img.save()
                    exported.append(os.path.basename(dst))
                except Exception as ee:
                    print("WARN: could not export image", img.name, "->", ee)
        except Exception as e:
            print("WARN: failed to process image", getattr(img, "name", "<noname>"), "->", e)

    with open(os.path.join(out_dir, "textures_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"exported": exported, "referenced": referenced}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()