# 3D Indoor Localization System with RSSI and Sensor Fusion

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Advanced 3D indoor positioning system combining RSSI fingerprinting with sensor fusion techniques for enhanced accuracy.

## Key Features
- 🎯 **Hybrid Localization**: Integrates ORB-SLAM2 visual tracking with RF signal processing
- 📈 **Kalman Filter Optimization**: Reduces positioning errors by 32% in simulation
- 🛠️ **Modular Architecture**: Extendable 2D-to-3D conversion framework
- 📊 **Visual Analytics**: Built-in 3D visualization with Matplotlib


### Installation

1. Clone repository
```sh
git clone https://github.com/GRay525/Enhanced-Indoor-Localization-/
cd main_v3.py
```
2. Install dependencies
```sh
pip install -r requirements.txt
```
3. Core dependencies
```sh
numpy==1.21.0
scipy==1.7.0
matplotlib==3.4.0
ecos==2.0.7.post1
Usage Example
python
from localization import joint_estimation, do_plot
```
4. Generate simulation data
```sh
xyz_seat, xyz_target, xyz_anchor, *_ = for_test_generate_simu_data(use_12_anchors=True)
```
5. Run localization algorithm
```sh
xyz_est, gamma_est, p0_est, *_ = joint_estimation(
    rss_meas, 
    xyz_anchor,
    max_iter=15,
    max_gain_err_db=9,
    min_z=0, 
    max_z=2
)
```
6. Visualize results
```sh
do_plot(xyz_est, xyz_anchor)
Algorithm Overview
Core Components
Sensor Fusion Architecture
```


###License
Distributed under MIT License. See LICENSE for details.
