"""Prepare target-FPS recorded RGB-D videos in the flat layout WiLoR expects."""
from __future__ import annotations
import argparse, json, os, subprocess
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--camera',required=True); p.add_argument('--target-fps',type=int,required=True); a=p.parse_args()
    info=json.loads((a.source/'meta/info.json').read_text()); source_fps=int(round(float(info['fps'])))
    if source_fps%a.target_fps: raise ValueError(f'{source_fps=} is not divisible by {a.target_fps=}')
    stride=source_fps//a.target_fps; episodes=[json.loads(x) for x in (a.source/'meta/episodes.jsonl').read_text().splitlines()]
    for suffix,ext in [('color','mp4'),('transformed_depth','mkv')]:
        key=f'observation.images.{a.camera}.{suffix}'; outdir=a.output/key; outdir.mkdir(parents=True,exist_ok=True)
        if key not in info['features']: raise KeyError(f'{key} is not present in source dataset')
        for episode in episodes:
            i=int(episode['episode_index']); chunk=i//int(info.get('chunks_size',1000)); rel=info['video_path'].format(episode_chunk=chunk,episode_index=i,video_key=key); src=a.source/rel
            if not src.exists() and suffix=='transformed_depth': src=src.with_suffix('.mkv')
            if not src.exists(): raise FileNotFoundError(src)
            dst=outdir/f'episode_{i:06d}.{ext}'
            if dst.exists(): continue
            if dst.is_symlink(): dst.unlink()
            if stride==1: os.symlink(src.resolve(),dst)
            else:
                codec=['-c:v','ffv1'] if ext=='mkv' else []
                subprocess.run(['/usr/bin/ffmpeg','-loglevel','error','-i',str(src),'-vf',f'select=not(mod(n\\,{stride})),setpts=N/({a.target_fps}*TB)','-an','-r',str(a.target_fps),'-fps_mode','cfr',*codec,str(dst)],check=True)
            print(dst)
if __name__=='__main__': main()
