
This is a version where Sam2 (https://codeload.github.com/facebookresearch/sam2) is converted from Python to C# using Doubao, retaining the original code structure and naming.

Reference the TorchSharp.dll and TorchVision.dll files from the DLL directory.

Copy LibTorshSharp.dll from the DLL directory to the runtime directory Runtimes//win-x64/native

This program requires the support of CUDA 13.0 and PyTorch 2.10.0. You need to download LibTorch from the following website: https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-2.10.0%2Bcu130.zip
After unzipping, copy all DLL files in the Lib directory to the runtime directory Runtimes//win-x64/native.

Download https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

Download https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt

Download https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_Small.pt

<img width="1103" height="667" alt="屏幕截图 2026-02-16 180822" src="https://github.com/user-attachments/assets/664fbbb0-4d62-4071-b5f1-19a6313bd906" />

Move KEY POINTS Segmentate different rigens.

<img width="1091" height="671" alt="屏幕截图 2026-02-16 180758" src="https://github.com/user-attachments/assets/035f05af-9fb5-41c9-82a6-b6b76c3e977b" />
...
