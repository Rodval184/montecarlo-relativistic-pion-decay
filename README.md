# Monte Carlo: Relativistic Pion Decay Survival Fraction

Numerical simulation and data analysis of a physical system using Python and scientific computing tools.

## 📌 Description
This project uses Monte Carlo methods to simulate **pion decay** in the laboratory frame.
Decay times are sampled from an exponential distribution and combined with relativistic time dilation to estimate the fraction of pions that survive up to a detector located at distance **L**.

Two scenarios are considered:
- **(a) Fixed kinetic energy** (monoenergetic beam)
- **(b) Kinetic energy sampled from a Gaussian distribution** (truncated at K > 0)

## 🛠️ Tools
- Python
- NumPy

## 📊 Methodology
- Compute relativistic factors (β, γ) from kinetic energy K
- Lab-frame lifetime: τ_lab = γ τ
- Sample decay times: t = -τ_lab ln(r), r ~ U(0,1)
- Distance traveled: d = v t
- Count survivors: d ≥ L

## ▶️ How to run
```bash
pip install -r requirements.txt
python src/main.py
