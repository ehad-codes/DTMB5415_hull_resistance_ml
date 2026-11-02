# Design-Space Dimensionality Reduction Benchmark Dataset – DTMB 5415 Hull

## Overview

This dataset provides a benchmark database for the evaluation and comparison of design-space dimensionality reduction methods in hydrodynamic shape optimization problems.

The dataset consists of **9,001 configurations** of the DTMB 5415 hull geometry generated through Monte Carlo sampling of a nonlinear parametric model based on global 3D shape modifications.

Each configuration is described by **27 design variables**, which define perturbations of the reference hull geometry. The hull surface is discretized using **2,250 points**, stored as flattened coordinate vectors.

---

## Data Content

Each configuration includes:

- discretized hull geometry coordinates (2,250 × x, y, z points)  
- normalized design variables defining the parametric model  
- pressure distribution on the hull evaluated at 2,250 surface points (for multiple speeds)  
- wave elevation field evaluated on a structured grid (1,500 points)  
- integrated hydrodynamic quantities:
  - wave resistance coefficient (Cw)  
  - RMS vertical acceleration at the bridge (az)  
  - RMS pitch motion (xi5)

---

## Physical Modeling

Hydrodynamic quantities were computed using low-fidelity solvers:

- **WARP** (linear potential flow) for calm-water resistance  
- **SMP** (linear strip theory) for seakeeping  

Simulation conditions:

- head waves  
- sea state 5  
- Bretschneider spectrum  
- Froude numbers:
  - Fr = 0.25  
  - Fr = 0.33  
  - Fr = 0.41  

---

## Scaling and Reference Conditions

The dataset combines **model-scale and full-scale representations**:

- Geometry is provided at **full scale**  
  - Length between perpendiculars: **Lpp = 142 m**

- Seakeeping quantities (motions and accelerations) are computed at **full scale**

- Calm-water quantities (resistance, pressure, wave elevation) are computed at **model scale**:
  - Lpp = 5.720 m

Reference model-scale parameters:

- Displacement: 0.549 tonnes  
- Beam: 0.760 m  
- Draft: 0.248 m  
- Longitudinal center of gravity: 2.884 m  
- Vertical center of gravity: 0.056 m  
- Water density: 998.5 kg/m³  
- Kinematic viscosity: 1.09 × 10⁻⁶ m²/s  
- Gravity acceleration: 9.803 m/s²  

---

## Dataset Files

The dataset includes:

- `DTMB5415_database.mat` → MATLAB file containing all matrices  
- `metadata.txt` → description of matrix structure  
- `README.md` → dataset documentation  

---

## Dataset Structure

The `.mat` file contains separate matrices for geometry, variables, and physical quantities.  
These must be assembled to construct the full data matrix used in dimensionality reduction workflows.

---

## Dataset Statistics

- total configurations: 9001  
- baseline configuration: 1 (DTMB 5415 reference hull)  
- sampled configurations: 9000  
- invalid simulations: 0  

All configurations included in the dataset correspond to **physically valid simulations**.  

During the Monte Carlo generation process, a larger number of samples was initially produced, and only configurations yielding valid physical responses were retained in the final dataset.

---

## Intended Use

This dataset is intended for:

- benchmarking dimensionality reduction methods  
- testing PME / PI-PME / PD-PME  
- surrogate modeling  
- design-space analysis  

---

## Scientific Context

The dataset follows a physics-informed representation of the design space, where geometric, parametric, and physical information are combined consistently within a unified framework.

---

## References

- Serani, A., Palma, G., Wackers, J., Quagliarella, D., Gaggero, S., & Diez, M. (2025). Extending parametric model embedding with physical information for design-space dimensionality reduction in shape optimization. Engineering with Computers, 1-21.

- Serani, A., Stern, F., Campana, E. F., & Diez, M. (2022). Hull-form stochastic optimization via computational-cost reduction methods. Engineering with computers, 38(Suppl 3), 2245-2269.

---

## Notes

- Dataset generated with low-fidelity solvers → suitable for design-space exploration and dimensionality reduction  
- Not intended for final design validation  
