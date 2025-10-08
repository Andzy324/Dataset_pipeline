# cam_viz.py  ——  Camera poses visualizer for PyTorch3D FoVPerspectiveCameras
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math, numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import open3d as o3d
    HAVE_O3D = True
except Exception:
    HAVE_O3D = False

def _to_numpy(x):
    try:
        import torch
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def camera_centers_from_p3d(cameras) -> np.ndarray:
    """PyTorch3D (row-vector): X_cam = X_world @ R.T + T  →  C = -T @ R"""
    R = _to_numpy(cameras.R)   # (N,3,3)
    T = _to_numpy(cameras.T)   # (N,3)
    # PyTorch3D stores row-major rotations; real camera centers require R^T.
    return -np.einsum('ni,nji->nj', T, R)

def _bbox_pts_edges(bmin, bmax):
    x0,y0,z0 = bmin.tolist(); x1,y1,z1 = bmax.tolist()
    pts = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ], float)
    edges = np.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ], int)
    return pts, edges

def _mpl_equal_aspect(ax, X, Y, Z):
    mr = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*(X.max()+X.min()); Yb = 0.5*(Y.max()+Y.min()); Zb = 0.5*(Z.max()+Z.min())
    r = mr/2.0 if mr>0 else 1.0
    ax.set_xlim(Xb-r, Xb+r); ax.set_ylim(Yb-r, Yb+r); ax.set_zlim(Zb-r, Zb+r)

def _basis_from_forward(fwd):
    f = fwd / (np.linalg.norm(fwd)+1e-8)
    up0 = np.array([0.0,0.0,1.0])
    if abs(np.dot(f, up0)) > 0.99: up0 = np.array([0.0,1.0,0.0])
    r = np.cross(f, up0);  r /= (np.linalg.norm(r)+1e-8)
    u = np.cross(r, f);    u /= (np.linalg.norm(u)+1e-8)
    return r,u,f

def _draw_frustum(ax, C, look_dir, fov_deg, aspect, near):
    r,u,f = _basis_from_forward(look_dir)
    half_h = math.tan(math.radians(fov_deg)/2.0)*near
    half_w = half_h*aspect
    cn = C + f*near
    quad = np.stack([
        cn + (-half_w)*r + (-half_h)*u,
        cn + ( half_w)*r + (-half_h)*u,
        cn + ( half_w)*r + ( half_h)*u,
        cn + (-half_w)*r + ( half_h)*u,
    ],0)
    for p in quad:
        ax.plot([C[0],p[0]],[C[1],p[1]],[C[2],p[2]], lw=0.7, alpha=0.65)
    for i in range(4):
        p0, p1 = quad[i], quad[(i+1)%4]
        ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]], lw=0.7, alpha=0.65)

def _get_fov_ar(cams, i, fov_fallback=50.0, ar_fallback=1.0):
    fov = getattr(cams, "fov", None)
    ar  = getattr(cams, "aspect_ratio", None)
    fovi = float(_to_numpy(fov)[i]) if fov is not None else float(fov_fallback)
    ari  = float(_to_numpy(ar)[i])  if ar  is not None else float(ar_fallback)
    return fovi, ari

def _print_stats(C, center, meta=None):
    d = np.linalg.norm(C-center[None,:], axis=1)
    print(f"[cam_viz] N={len(C)} | dist mean={d.mean():.4g}, std={d.std():.4g}, min={d.min():.4g}, max={d.max():.4g}")
    if meta:
        off=0
        for m in meta:
            n=int(m.get('num',0)); name=m.get('name','?')
            if n<=1: off+=n; continue
            az=m.get('azims'); el=m.get('elevs')
            msg=f"  - ring '{name}': n={n}"
            if az is not None:
                az=np.asarray(az,float)
                dif=np.mod(np.diff(np.r_[az, az[0]+360.0]),360.0)
                msg+=f" | Δaz mean={dif.mean():.3f}° (≈{360.0/n:.3f}°), std={dif.std():.3f}°"
            if el is not None:
                el=np.asarray(el,float); msg+=f" | elev first={el[0]:.3f}°"
            print(msg); off+=n

def viz_mpl(cameras, center, bmin, bmax, out_dir: Union[str,Path],
            meta=None, connect_by_ring=True, draw_frustum=False, max_frustums=50,
            view_elev=25.0, view_azim=-60.0):
    C = camera_centers_from_p3d(cameras)
    center=_to_numpy(center).reshape(3); bmin=_to_numpy(bmin).reshape(3); bmax=_to_numpy(bmax).reshape(3)
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _print_stats(C, center, meta)

    fig=plt.figure(figsize=(7,6), dpi=140); ax=fig.add_subplot(111,projection='3d'); ax.set_title("Camera Poses")
    # bbox
    pts,edges=_bbox_pts_edges(bmin,bmax)
    for e in edges:
        p0,p1=pts[e[0]],pts[e[1]]
        ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]], lw=1.0, alpha=0.8)
    ax.scatter([center[0]],[center[1]],[center[2]], s=35, marker='x')
    # cameras
    ax.scatter(C[:,0],C[:,1],C[:,2], s=10, alpha=0.9)
    # look-at arrows
    v = center[None,:]-C; L=np.linalg.norm(v,axis=1,keepdims=True)+1e-8; U=v/L
    P2 = C + 0.12*L*U
    for i in range(len(C)):
        ax.plot([C[i,0],P2[i,0]],[C[i,1],P2[i,1]],[C[i,2],P2[i,2]], lw=0.6, alpha=0.55)
    # connect rings
    if connect_by_ring and meta is not None:
        off=0
        for m in meta:
            n=int(m.get("num",0))
            if n>1:
                Ci=C[off:off+n]
                ax.plot(Ci[:,0],Ci[:,1],Ci[:,2], lw=0.8, alpha=0.7)
            off+=n
    # frustums
    if draw_frustum:
        scene_scale=max(1e-6, np.linalg.norm(bmax-bmin))
        near=0.12*scene_scale
        idx=np.linspace(0,len(C)-1, num=min(len(C),int(max_frustums)), dtype=int)
        for i in idx:
            fov,ar=_get_fov_ar(cameras,i)
            _draw_frustum(ax, C[i], center-C[i], fov, ar, near)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    _mpl_equal_aspect(ax, np.r_[bmin[0],bmax[0],C[:,0]],
                         np.r_[bmin[1],bmax[1],C[:,1]],
                         np.r_[bmin[2],bmax[2],C[:,2]])
    ax.view_init(elev=float(view_elev), azim=float(view_azim))
    fig.tight_layout()
    out_png = out_dir/"poses_3d.png"; fig.savefig(str(out_png)); plt.close(fig)
    return {"poses": str(out_png)}

def viz_open3d(cameras, center, bmin, bmax, out_dir: Union[str,Path], show_window=False):
    if not HAVE_O3D:
        print("[cam_viz] Open3D not available; skip."); return {}
    C = camera_centers_from_p3d(cameras)
    center=_to_numpy(center).reshape(3); bmin=_to_numpy(bmin).reshape(3); bmax=_to_numpy(bmax).reshape(3)
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # build geoms
    geoms=[]
    pts,edges=_bbox_pts_edges(bmin,bmax)
    ls=o3d.geometry.LineSet(); ls.points=o3d.utility.Vector3dVector(pts); ls.lines=o3d.utility.Vector2iVector(edges)
    geoms.append(ls)
    scale=max(1e-3, 0.01*np.linalg.norm(bmax-bmin))
    for p in C:
        s=o3d.geometry.TriangleMesh.create_sphere(radius=scale); s.translate(p); s.compute_vertex_normals(); geoms.append(s)
    rays_pts=[]; rays_edges=[]
    for i,p in enumerate(C): rays_pts+=[p,center]; rays_edges.append([2*i,2*i+1])
    ray=o3d.geometry.LineSet(); ray.points=o3d.utility.Vector3dVector(np.asarray(rays_pts)); ray.lines=o3d.utility.Vector2iVector(np.asarray(rays_edges,int))
    geoms.append(ray)
    if show_window:
        o3d.visualization.draw_geometries(geoms); return {}
    # offscreen
    out_img = out_dir/"poses_o3d.png"
    try:
        ren=o3d.visualization.rendering.OffscreenRenderer(1280,960)
        mat=o3d.visualization.rendering.MaterialRecord(); mat.shader="defaultLit"
        for k,g in enumerate(geoms): ren.scene.add_geometry(f"g{k}", g, mat)
        ctr=0.5*(bmin+bmax); extent=np.linalg.norm(bmax-bmin); eye=ctr+np.array([extent,extent,extent])*0.8+1e-3
        ren.scene.camera.look_at(ctr, eye, np.array([0,0,1.0]))
        img=ren.render_to_image(); o3d.io.write_image(str(out_img), img)
        return {"poses": str(out_img)}
    except Exception as e:
        print(f"[cam_viz] OffscreenRenderer failed: {type(e).__name__}: {e}"); return {}

def visualize_cameras(cameras, center, bmin, bmax, out_dir: Union[str,Path],
                      meta: Optional[List[Dict]]=None, backend:str="mpl",
                      draw_frustum: bool=False, max_frustums:int=50, show:bool=False):
    if backend.lower()=="open3d":
        return viz_open3d(cameras, center, bmin, bmax, out_dir, show_window=show)
    return viz_mpl(cameras, center, bmin, bmax, out_dir, meta=meta,
                   draw_frustum=draw_frustum, max_frustums=max_frustums)
