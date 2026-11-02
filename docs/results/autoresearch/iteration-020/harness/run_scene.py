"""Supervise one full decoder integration run under the shared device lock."""
import argparse,fcntl,json,os,pathlib,sys,time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--candidate-root",type=pathlib.Path,required=True)
p.add_argument("--reference-dir",type=pathlib.Path,required=True)
p.add_argument("--output-dir",type=pathlib.Path,required=True)
p.add_argument("--warm-repeats",type=int,default=2)
p.add_argument("--algorithm",choices=("digit_explicit_gemm","digit_implicit_gemm"),required=True)
p.add_argument("--topology-policy",choices=("baseline","stage"),required=True)
a=p.parse_args()
if a.output_dir.exists():p.error("fresh output directory required")
a.output_dir.parent.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(a.candidate_root))
from benchmarks.tenstorrent.run import monitor
from benchmarks.tenstorrent.common import sha256,write_json
script=pathlib.Path(__file__).with_name("digit_scene_probe.py")
record={"status":"waiting_for_device","candidate_root":str(a.candidate_root),"algorithm":a.algorithm,"topology_policy":a.topology_policy,"script_sha256":sha256(script),"whole_worker_timeout_seconds":900}
runner=a.output_dir.with_suffix(".runner.json");write_json(runner,record)
with open("/tmp/genrecon-tt-pjrt.lock","a") as lock:
 start=time.monotonic();fcntl.flock(lock,fcntl.LOCK_EX)
 record.update(status="running",lock_wait_seconds=time.monotonic()-start);write_json(runner,record)
 with a.output_dir.with_suffix(".log").open("w") as log:
  r=monitor([sys.executable,str(script),"--reference-dir",str(a.reference_dir),"--output-dir",str(a.output_dir),"--precision","bf16","--refresh-fixed-shape","--warm-repeats",str(a.warm_repeats)],log=log,report_path=a.output_dir/"result.json",compile_timeout=900,execution_timeout=900,env=dict(os.environ,FLEXGEMM_CANDIDATE_ROOT=str(a.candidate_root),FLEXGEMM_SCENE_LOCKED="1",FLEXGEMM_DIGIT_ALGORITHM=a.algorithm,FLEXGEMM_TOPOLOGY_POLICY=a.topology_policy))
 record.update(r);record["status"]="timeout" if r["timeout"] else "complete";write_json(runner,record);print(json.dumps(record))
raise SystemExit(record["returncode"])
