"""Create a complete local LeRobot v2.1 dataset at a lower integer-divisor FPS."""
from __future__ import annotations
import argparse, json, shutil, subprocess
from pathlib import Path
import numpy as np, pandas as pd
FFMPEG='/usr/bin/ffmpeg'; FFPROBE='/usr/bin/ffprobe'

def lines(path: Path): return [json.loads(x) for x in path.read_text().splitlines()]
def dump(path: Path, rows): path.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in rows))

def video_frames(path: Path) -> int:
    out=subprocess.check_output([FFPROBE,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','default=nokey=1:noprint_wrappers=1',str(path)],text=True).strip()
    return int(out)

def column_stats(series: pd.Series) -> dict:
    first=series.iloc[0]
    values=np.stack(series.to_numpy()) if isinstance(first,(list,np.ndarray)) else series.to_numpy()[:,None]
    return {'min':values.min(0).tolist(),'max':values.max(0).tolist(),'mean':values.mean(0).tolist(),'std':values.std(0).tolist(),'count':[len(values)]}

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--target-fps',type=int,required=True); a=p.parse_args()
    if (a.output/'.complete').exists(): raise FileExistsError(f'{a.output} is already complete')
    info=json.loads((a.source/'meta/info.json').read_text()); source_fps=int(round(float(info['fps'])))
    if source_fps%a.target_fps: raise ValueError(f'source FPS {source_fps} must be divisible by target FPS {a.target_fps}')
    stride=source_fps//a.target_fps; episodes=lines(a.source/'meta/episodes.jsonl'); old_stats={x['episode_index']:x for x in lines(a.source/'meta/episodes_stats.jsonl')}
    (a.output/'meta').mkdir(parents=True,exist_ok=True); shutil.copy2(a.source/'meta/tasks.jsonl',a.output/'meta/tasks.jsonl')
    new_episodes=[]; new_stats=[]; global_index=0; video_keys=[k for k,v in info['features'].items() if v.get('dtype')=='video']
    for episode in episodes:
        ep=int(episode['episode_index']); chunk=ep//int(info.get('chunks_size',1000)); data_rel=info['data_path'].format(episode_chunk=chunk,episode_index=ep)
        df=pd.read_parquet(a.source/data_rel).iloc[::stride].copy().reset_index(drop=True); n=len(df)
        df['frame_index']=np.arange(n,dtype=np.int64); df['timestamp']=np.arange(n,dtype=np.float32)/a.target_fps; df['index']=np.arange(global_index,global_index+n,dtype=np.int64); global_index+=n
        dst_data=a.output/data_rel; dst_data.parent.mkdir(parents=True,exist_ok=True); df.to_parquet(dst_data,index=False)
        for key in video_keys:
            rel=Path(info['video_path'].format(episode_chunk=chunk,episode_index=ep,video_key=key)); src=a.source/rel
            if not src.exists() and src.suffix=='.mp4': src=src.with_suffix('.mkv')
            dst=a.output/rel.with_suffix(src.suffix); dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():
                try:
                    if video_frames(dst)==n: continue
                except Exception:
                    pass
                dst.unlink()
            if stride==1: shutil.copy2(src,dst)
            else:
                codec=['-c:v','ffv1'] if src.suffix=='.mkv' else ['-c:v','libx264','-crf','18']
                subprocess.run([FFMPEG,'-loglevel','error','-i',str(src),'-vf',f'select=not(mod(n\\,{stride})),setpts=N/({a.target_fps}*TB)','-an','-r',str(a.target_fps),'-fps_mode','cfr',*codec,str(dst)],check=True)
            actual=video_frames(dst)
            if actual!=n: raise ValueError(f'{dst}: video frames={actual}, parquet rows={n}')
        updated={**episode,'length':n}; new_episodes.append(updated)
        stats=old_stats[ep]
        for value in stats['stats'].values(): value['count']=[n]
        for key in df.columns:
            stats['stats'][key]=column_stats(df[key])
        new_stats.append(stats); print(f'episode {ep}: {len(pd.read_parquet(a.source/data_rel))} -> {n}')
    info.update(fps=a.target_fps,total_frames=global_index,total_episodes=len(new_episodes),total_videos=len(new_episodes)*len(video_keys),total_chunks=max(1,(len(new_episodes)+int(info.get('chunks_size',1000))-1)//int(info.get('chunks_size',1000))))
    (a.output/'meta/info.json').write_text(json.dumps(info,indent=4)+'\n'); dump(a.output/'meta/episodes.jsonl',new_episodes); dump(a.output/'meta/episodes_stats.jsonl',new_stats)
    (a.output/'.complete').write_text(f'fps={a.target_fps}\nepisodes={len(new_episodes)}\nframes={global_index}\n')
    print(f'OK: episodes={len(new_episodes)} frames={global_index} fps={a.target_fps} output={a.output}')
if __name__=='__main__': main()
