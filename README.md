这是使用豆包将Sam2(https://codeload.github.com/facebookresearch/sam2)从Python转换为C#的版本，保留原代码结构及命名；

引用TorchSharp.dll、TorchVision.dll文件从DLL目录;

拷贝LibTorshSharp.dll至运行目录Runtimes//win-x64/native目录从DLL目录

本程序需CUDA13.0、PyTorch2.10.0支持，需要下载LibTorch从以下网站： https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-2.10.0%2Bcu130.zip
解压后，将Lib目录下DLL文件全部拷贝至运行目录Runtimes//win-x64/native目录；

下载https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

下载https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt

下载https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_Small.pt

移动KEY POINT 分割不同区域