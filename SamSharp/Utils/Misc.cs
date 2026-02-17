using OpenCvSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace Sam2Sharp.Utils
{

    public static class SdpaSettings
    {
        public static (bool oldGpu, bool useFlashAttn, bool mathKernelOn) GetSdpaSettings()
        {
            if (torch.cuda.is_available())
            {
                // TorchSharp 没有直接获取 CUDA 设备属性的 API，通常只能假设新卡或通过环境变量/配置判断
                // 这里保守假设为新卡（Ampere 8.0+），如有更好的 API 可替换
                bool oldGpu = false;
                bool useFlashAttn = true;

                var versionParts = torch.__version__.Split('.').Take(2).Select(int.Parse).ToArray();
                bool mathKernelOn = (versionParts[0] < 2 || (versionParts[0] == 2 && versionParts[1] < 2)) || !useFlashAttn;

                if (versionParts[0] < 2 || (versionParts[0] == 2 && versionParts[1] < 2))
                {
                    Console.WriteLine($"Warning: You are using PyTorch {torch.__version__} without Flash Attention v2 support. Consider upgrading to PyTorch 2.2+ for better performance.");
                }

                return (oldGpu, useFlashAttn, mathKernelOn);
            }
            else
            {
                return (true, false, true);
            }
        }
    }

    public static class MaskProcessing
    {
        public static (Tensor labels, Tensor counts) GetConnectedComponents(Tensor mask)
        {
            // 注意：这里需要C++扩展的C#绑定，实际实现需根据底层库调整
            using (var scope = torch.NewDisposeScope())
            {
                var uint8Mask = mask.to(torch.uint8).contiguous();
                // 假设SAM2库提供了相应的C#绑定
                var (labels, counts) = SAM2Native.GetConnectedComponents(uint8Mask);
                return (labels.MoveToOuterDisposeScope(), counts.MoveToOuterDisposeScope());
            }
        }


        public static Tensor FillHolesInMaskScores(Tensor mask, double maxArea)
        {
            using (var scope = torch.NewDisposeScope())
            {
                if (maxArea <= 0)
                    throw new ArgumentException("max_area must be positive");

                var inputMask = mask;
                try
                {
                    var (labels, areas) = GetConnectedComponents(mask <= 0);
                    var isHole = (labels > 0) & (areas <= maxArea);
                    var result = torch.where(isHole, 0.1, mask);
                    return result.MoveToOuterDisposeScope();
                }
                catch (Exception e)
                {
                    Console.WriteLine($"Warning: {e.Message}\nSkipping post-processing step...");
                    return inputMask.MoveToOuterDisposeScope();
                }
            }
        }
    }

    public class AsyncVideoFrameLoader
    {
        private readonly string[] _imgPaths;
        private readonly int _imageSize;
        private readonly bool _offloadVideoToCpu;
        private readonly Tensor _imgMean;
        private readonly Tensor _imgStd;
        private readonly Device _computeDevice;
        private Tensor[] _images;
        private Exception _exception;
        public int VideoHeight { get; private set; }
        public int VideoWidth { get; private set; }
        private Thread _loadThread;

        public AsyncVideoFrameLoader(string[] imgPaths, int imageSize, bool offloadVideoToCpu,
                                    Tensor imgMean, Tensor imgStd, Device computeDevice)
        {
            _imgPaths = imgPaths;
            _imageSize = imageSize;
            _offloadVideoToCpu = offloadVideoToCpu;
            _imgMean = imgMean;
            _imgStd = imgStd;
            _computeDevice = computeDevice;
            _images = new Tensor[imgPaths.Length];
            _exception = null;

            // 加载第一帧
            GetItem(0);

            // 异步加载剩余帧
            _loadThread = new Thread(LoadFrames);
            _loadThread.IsBackground = true;
            _loadThread.Start();
        }

        public Tensor GetItem(int index)
        {
            if (_exception != null)
                throw new InvalidOperationException("Failure in frame loading thread", _exception);

            if (_images[index] is not null)
                return _images[index];

            var (img, videoHeight, videoWidth) = LoadImgAsTensor(_imgPaths[index], _imageSize);
            VideoHeight = videoHeight;
            VideoWidth = videoWidth;

            // 归一化
            img = img.sub(_imgMean).div(_imgStd);

            if (!_offloadVideoToCpu)
                img = img.to(_computeDevice, non_blocking: true);

            _images[index] = img;
            return img;
        }

        private void LoadFrames()
        {
            try
            {
                for (int i = 0; i < _images.Length; i++)
                {
                    GetItem(i);
                    Console.WriteLine($"Loaded frame {i + 1}/{_images.Length}");
                }
            }
            catch (Exception e)
            {
                _exception = e;
            }
        }

        private (Tensor, int, int) LoadImgAsTensor(string imgPath, int imageSize)
        {
            using (var mat = Cv2.ImRead(imgPath))
            {
                if (mat.Empty())
                    throw new FileNotFoundException("Could not load image", imgPath);

                // 转换为RGB并调整大小
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2RGB);
                Cv2.Resize(mat, mat, new OpenCvSharp.Size(imageSize, imageSize));

                // 转换为Tensor
                var imgArray = new float[imageSize * imageSize * 3];
                mat.GetArray(out imgArray);
                var img = torch.tensor(imgArray, dtype: torch.float32)
                              .reshape(3, imageSize, imageSize)
                              .div(255.0f);

                return (img, mat.Rows, mat.Cols);
            }
        }

        public int Count => _images.Length;
    }

    public static class VideoLoader
    {
        public static (object frames, int height, int width) LoadVideoFrames(string videoPath, int imageSize,
                                                                           bool offloadVideoToCpu,
                                                                           (double, double, double)? imgMean = null,
                                                                           (double, double, double)? imgStd = null,
                                                                           bool asyncLoadingFrames = false,
                                                                           Device computeDevice = null)
        {
            computeDevice ??= torch.CUDA;
            var mean = imgMean ?? (0.485, 0.456, 0.406);
            var std = imgStd ?? (0.229, 0.224, 0.225);

            if (Directory.Exists(videoPath))
            {
                return LoadVideoFramesFromJpgImages(videoPath, imageSize, offloadVideoToCpu,
                                                  mean, std, asyncLoadingFrames, computeDevice);
            }
            else if (File.Exists(videoPath) && Path.GetExtension(videoPath).Equals(".mp4", StringComparison.OrdinalIgnoreCase))
            {
                return LoadVideoFramesFromVideoFile(videoPath, imageSize, offloadVideoToCpu,
                                                  mean, std, computeDevice);
            }
            else
            {
                throw new NotImplementedException("Only MP4 video and JPEG folder are supported");
            }
        }

        public static (object frames, int height, int width) LoadVideoFramesFromJpgImages(string videoPath,
                                                                                        int imageSize,
                                                                                        bool offloadVideoToCpu,
                                                                                        (double, double, double) imgMean,
                                                                                        (double, double, double) imgStd,
                                                                                        bool asyncLoadingFrames,
                                                                                        Device computeDevice)
        {
            var jpgFiles = Directory.EnumerateFiles(videoPath)
                                   .Where(f => new[] { ".jpg", ".jpeg" }.Contains(Path.GetExtension(f).ToLower()))
                                   .OrderBy(f => int.Parse(Path.GetFileNameWithoutExtension(f)))
                                   .ToArray();

            if (jpgFiles.Length == 0)
                throw new DirectoryNotFoundException("No JPEG images found in directory");

            var meanTensor = torch.tensor(new[] { imgMean.Item1, imgMean.Item2, imgMean.Item3 }, dtype: torch.float32)
                                 .unsqueeze(1).unsqueeze(1);
            var stdTensor = torch.tensor(new[] { imgStd.Item1, imgStd.Item2, imgStd.Item3 }, dtype: torch.float32)
                                .unsqueeze(1).unsqueeze(1);

            if (asyncLoadingFrames)
            {
                var loader = new AsyncVideoFrameLoader(jpgFiles, imageSize, offloadVideoToCpu, meanTensor, stdTensor, computeDevice);
                return (loader, loader.VideoHeight, loader.VideoWidth);
            }
            else
            {
                int numFrames = jpgFiles.Length;
                var images = torch.zeros(numFrames, 3, imageSize, imageSize, dtype: torch.float32);
                int videoHeight = 0, videoWidth = 0;

                for (int i = 0; i < numFrames; i++)
                {
                    using (var mat = Cv2.ImRead(jpgFiles[i]))
                    {
                        Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2RGB);
                        Cv2.Resize(mat, mat, new OpenCvSharp.Size(imageSize, imageSize));

                        videoHeight = mat.Rows;
                        videoWidth = mat.Cols;

                        var imgArray = new float[imageSize * imageSize * 3];
                        mat.GetArray(out imgArray);
                        images[i] = torch.tensor(imgArray, dtype: torch.float32)
                                         .reshape(3, imageSize, imageSize)
                                         .div(255.0f);
                    }
                }

                if (!offloadVideoToCpu)
                {
                    images = images.to(computeDevice);
                    meanTensor = meanTensor.to(computeDevice);
                    stdTensor = stdTensor.to(computeDevice);
                }

                images = images.sub(meanTensor).div(stdTensor);
                return (images, videoHeight, videoWidth);
            }
        }

        public static (Tensor frames, int height, int width) LoadVideoFramesFromVideoFile(string videoPath,
                                                                                        int imageSize,
                                                                                        bool offloadVideoToCpu,
                                                                                        (double, double, double) imgMean,
                                                                                        (double, double, double) imgStd,
                                                                                        Device computeDevice)
        {
            // 注意：实际实现需要视频解码库（如FFmpeg绑定）
            throw new NotImplementedException("Video file loading requires FFmpeg bindings");
        }

        public static (Tensor points, Tensor labels) ConcatPoints(Tensor oldPoints, Tensor oldLabels, Tensor newPoints, Tensor newLabels)
        {
            using (var scope = torch.NewDisposeScope())
            {
                if (oldPoints is null || oldLabels is null)
                {
                    return (newPoints.MoveToOuterDisposeScope(), newLabels.MoveToOuterDisposeScope());
                }
                else
                {
                    var combinedPoints = torch.cat(new[] { oldPoints, newPoints }, dim: 1);
                    var combinedLabels = torch.cat(new[] { oldLabels, newLabels }, dim: 1);
                    return (combinedPoints.MoveToOuterDisposeScope(), combinedLabels.MoveToOuterDisposeScope());
                }
            }
        }
    }

    // 假设的C++扩展绑定（实际需根据库实现）
    internal static class SAM2Native
    {
        [System.Runtime.InteropServices.DllImport("sam2")]
        public static extern (Tensor, Tensor) GetConnectedComponents(Tensor mask);
    }
}
