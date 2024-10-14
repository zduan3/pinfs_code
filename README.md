# Physics-Informed Neural Fields with Neural Implicit Surface for Fluid Reconstruction
The pytorch implementation of the paper "Physics-Informed Neural Fields with Neural Implicit Surface for Fluid Reconstruction", accepted by [Pacific Graphics 2024](https://pg2024.hsu.edu.cn/#/program).

# Run

The code is tested with Python 3.10, PyTorch 2.0 (with CUDA 11.8)

```bash
./download_data.sh

# reconstruct the Sphere scene using hybrid model
python run_pinf.py -c configs/sphere_neus.txt

# render test with the trained model
python run_pinf.py -c configs/sphere_neus.txt --render_only
```