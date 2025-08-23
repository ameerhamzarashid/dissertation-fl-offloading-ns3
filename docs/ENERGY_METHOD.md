# Energy-Centric Method

**Objective:** Minimize device-side energy while maintaining acceptable latency and accuracy.

## Reward
We use a weighted negative cost:
\[
r = - \big( w_l \cdot \tilde{L} + w_e \cdot \tilde{E} \big)
\]
where \( \tilde{L}, \tilde{E} \) are normalized latency (ms) and energy (J). Min–max is tracked online.

- Default: \(w_l = 0.5\), \(w_e = 0.5\).
- Alternative: adjust weights to bias energy saving vs latency.

## Energy accounting
- **CPU energy** (UE): \(E_{cpu} \approx e_{cycle} \cdot \text{cycles}\).
- **Radio energy** (UE): \(E_{radio} \approx P_{tx} t_{tx} + P_{rx} t_{rx}\).
- Replace the helper with ns-3 EnergyModel for radio + CPU for higher fidelity.

## Communication efficiency
- **SFEA (Top‑k + error feedback)** reduces gradient payload by transmitting only the largest \(k\%\) entries and accumulating residuals locally for the next round.
- Expected outcome: order‑of‑magnitude drop in **comm bytes** and a corresponding decrease in **radio energy**.

## Experiments
- **Exp1 (Comm Efficiency):** Baseline (FedAvg) vs SFEA @ k=10%.
- **Exp2 (Trade‑off):** Report (latency, energy, reward) vs comm savings.
- **Exp3 (Scalability):** N∈{10,50}, k∈{20,50}.

