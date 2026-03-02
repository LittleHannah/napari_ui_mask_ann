"""
serve.py
--------
程序入口。修改 input_dir / initial_prefix 后直接运行:
    python serve.py
"""
import napari
from viewer import build_viewer

if __name__ == "__main__":
    input_dir = "/Users/xiaohanzhao/Projects/Dinov3_ST/napari/data/Visium_HD_Human_Kidney_FFPE/maskann_size-512"
    initial_prefix = "r0_c512"

    v = build_viewer(input_dir=input_dir, initial_prefix=initial_prefix)
    napari.run()
