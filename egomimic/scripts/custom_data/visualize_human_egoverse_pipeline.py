"""Validate alignment and save an RGB/keypoint/Zarr comparison image."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import cv2, numpy as np, zarr

EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]

def frame(path: Path, index: int) -> np.ndarray:
    cap=cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES,index); ok,bgr=cap.read(); cap.release()
    if not ok: raise IndexError(f"cannot read frame {index} from {path}")
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source',type=Path,required=True); p.add_argument('--keypoint-dir',type=Path,required=True); p.add_argument('--zarr-dir',type=Path,required=True); p.add_argument('--video-key',required=True); p.add_argument('--episode',type=int,default=0); p.add_argument('--frame',type=int,default=0); p.add_argument('--intrinsics',type=Path); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    candidates=sorted(a.zarr_dir.glob('*.zarr'))
    if not candidates: raise FileNotFoundError(f'no .zarr episodes under {a.zarr_dir}')
    expected=f'episode_{a.episode:06d}.mp4.keypoints3d.npy'
    matches=[path for path in candidates if Path(str(zarr.open_group(path,mode='r').attrs.get('source_keypoint_path',''))).name == expected]
    if len(matches) != 1: raise ValueError(f'expected one Zarr for {expected}, found {len(matches)}')
    zpath=matches[0]; zg=zarr.open_group(zpath,mode='r')
    info=json.loads((a.source/'meta/info.json').read_text()); stride=int(round(float(info['fps']))//zg.attrs['fps'])
    chunk=a.episode//int(info.get('chunks_size',1000)); rel=info['video_path'].format(episode_chunk=chunk,episode_index=a.episode,video_key=a.video_key)
    rgb=frame(a.source/rel,a.frame*stride); kp=np.load(a.keypoint_dir/f'episode_{a.episode:06d}.mp4.keypoints3d.npy')[a.frame]
    K=np.loadtxt(a.intrinsics or a.keypoint_dir/f"{a.video_key.removeprefix('observation.images.').removesuffix('.color')}_intrinsics.txt")
    uv=np.column_stack((K[0,0]*kp[:,0]/kp[:,2]+K[0,2],K[1,1]*kp[:,1]/kp[:,2]+K[1,2])).round().astype(int)
    overlay=rgb.copy()
    for i,j in EDGES: cv2.line(overlay,tuple(uv[i]),tuple(uv[j]),(0,255,0),3)
    for point in uv: cv2.circle(overlay,tuple(point),5,(255,0,0),-1)
    encoded=zg['images.front_1'][a.frame]
    while isinstance(encoded,np.ndarray) and encoded.ndim == 0: encoded=encoded.item()
    decoded=cv2.imdecode(np.frombuffer(encoded,dtype=np.uint8),cv2.IMREAD_COLOR)
    if decoded is None: raise ValueError(f'cannot decode images.front_1 frame {a.frame} from {zpath}')
    zimg=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
    zimg=cv2.resize(zimg,(overlay.shape[1],overlay.shape[0])); combined=np.concatenate((overlay,zimg),axis=1)
    a.output.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(a.output),cv2.cvtColor(combined,cv2.COLOR_RGB2BGR))
    print(f'OK episode={a.episode} frame={a.frame} keypoints={kp.shape} zarr_fps={zg.attrs["fps"]} pose_frame={zg.attrs["pose_frame"]} output={a.output}')

if __name__=='__main__': main()
