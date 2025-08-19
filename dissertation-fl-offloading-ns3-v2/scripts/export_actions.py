#!/usr/bin/env python3
import os, yaml, csv, numpy as np, torch, argparse
from src.dataset_synth import SyntheticMEC
from src.dqn import QNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run_baseline.yaml")
    ap.add_argument("--model", default=None, help="Path to global.pt")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--horizon", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    args = ap.parse_args()

    with open(args.config) as f: cfg = yaml.safe_load(f)
    K = cfg["data"]["servers"]; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = args.outdir or f"experiments/{cfg.get('experiment_name','run')}"
    os.makedirs(outdir, exist_ok=True)
    model_path = args.model or f"{outdir}/global.pt"
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    net = QNet(2+K, 1+K).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device)); net.eval()

    env = SyntheticMEC(cfg)
    with open(f"{outdir}/actions.csv","w", newline="") as f:
        w = csv.writer(f); w.writerow(["t","ue","action"])
        t = 0.0; steps = int(args.horizon/args.dt); I = cfg["data"]["users"]
        for _ in range(steps):
            for i in range(I):
                rates = env._rates_for_user(i)
                D, C  = env.sample_task()
                x = np.concatenate(([D, C], rates))
                with torch.no_grad():
                    a = int(net(torch.as_tensor(x, dtype=torch.float32, device=device)).argmax().item())
                w.writerow([round(t,3), i, a])
            t += args.dt
    print(f"[ok] wrote {outdir}/actions.csv")

if __name__ == "__main__":
    main()
