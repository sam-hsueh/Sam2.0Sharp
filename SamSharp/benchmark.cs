using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAM2.Benchmark
{
    class Program
    {
        // 配置参数（对应Python脚本的硬编码配置，可按需调整）
        private const string Sam2Checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt";
        private const string ModelCfg = "configs/sam2.1/sam2.1_hiera_b+.yaml";
        private const string VideoDir = "notebooks/videos/bedroom";
        private const int WarmUp = 5;
        private const int Runs = 25;
        private const bool Verbose = true;
        private const int AnnFrameIdx = 0;
        public const int AnnObjId = 1; // made public for reuse
        private static readonly (float, float) AnnPoint = (210f, 350f); // 标注点坐标

        // Simple no-op disposable for compatibility with removed autocast usage
        private class NoopDisposable : IDisposable { public static NoopDisposable Instance { get; } = new NoopDisposable(); public void Dispose() { } }

        static void Main(string[] args)
        {
            try
            {
                // 1. 环境检查与设备配置
                Console.WriteLine("=== 环境初始化 ===");
                if (!cuda.is_available())
                {
                    Console.WriteLine("错误：CUDA不可用，该脚本仅支持GPU推理（CUDA）。");
                    return;
                }
                var device = torch.device(DeviceType.CUDA);
                // Fallback: TorchSharp may not expose get_device_name; print basic info
                Console.WriteLine($"当前使用GPU设备：{(cuda.is_available() ? "CUDA" : "CPU")}");

                // 2. 性能优化配置（对应Python的混合精度+TF32加速）
                using (var autocast = NoopDisposable.Instance)
                {
                    // Ampere架构及以上（GPU算力≥8.0）启用TF32加速
                    try
                    {
                        var major = Sam2Sharp.Utils.CudaHelpers.TryGetDeviceMajor(0);
                        if (major.HasValue && major.Value >= 8)
                        {
                            torch.backends.cuda.matmul.allow_tf32 = true;
                            torch.backends.cudnn.allow_tf32 = true;
                            Console.WriteLine("已启用TF32精度加速（针对Ampere及以上架构GPU）。");
                        }
                    }
                    catch
                    {
                        // ignore if properties not available
                    }

                    // 3. 构建SAM2视频预测器（对应Python的build_sam2_video_predictor）
                    Console.WriteLine("\n=== 加载SAM2模型 ===");
                    var predictor = BuildSam2VideoPredictor(ModelCfg, Sam2Checkpoint, device, true);
                    Console.WriteLine("SAM2视频预测器加载完成。");

                    // 4. 视频帧准备与推理状态初始化
                    Console.WriteLine("\n=== 初始化视频与推理状态 ===");
                    var frameNames = GetSortedFrameNames(VideoDir);
                    int numFrames = frameNames.Count;
                    Console.WriteLine($"已加载视频帧数量：{numFrames}（来自目录：{VideoDir}）");

                    // 初始化推理状态（对应Python的predictor.init_state）
                    var inferenceState = predictor.InitState(VideoDir, 
                        offloadVideoToCpu: false, 
                        offloadStateToCpu: false, 
                        asyncLoadingFrames: false);
                    // omit torch.cuda.empty_cache() if not available
                    try { Sam2Sharp.Utils.CudaHelpers.TryEmptyCache(); } catch { }
                    Console.WriteLine("推理状态初始化完成。");

                    // 5. 设置初始标注点（对应Python的predictor.add_new_points_or_box）
                    Console.WriteLine("\n=== 设置初始目标标注 ===");
                    var points = CreateAnnotationPoints();
                    var labels = torch.tensor(new[] { 1 }, dtype: ScalarType.Int32, device: device); // 正标签（1）
                    var (_, outObjIds, outMaskLogits) = predictor.AddNewPointsOrBox(
                        inferenceState: inferenceState,
                        frameIdx: AnnFrameIdx,
                        objId: AnnObjId,
                        points: points,
                        labels: labels);
                    Console.WriteLine($"已在第{AnnFrameIdx}帧标注目标（ID：{AnnObjId}），标注点坐标：{AnnPoint}");

                    // 6. 基准测试主流程
                    Console.WriteLine("\n=== 开始性能基准测试 ===");
                    double totalElapsedSeconds = 0.0;
                    int validRunCount = 0;

                    using (var inferenceScope = torch.inference_mode()) // 推理模式（禁用梯度计算，优化性能）
                    using (var autocastInner = NoopDisposable.Instance) // 替代原始 autocast
                    {
                        for (int i = 0; i < Runs; i++)
                        {
                            if (Verbose)
                                Console.Write($"正在运行第 {i + 1}/{Runs} 次推理...");

                            // 计时开始
                            var stopwatch = Stopwatch.StartNew();

                            // 核心：视频目标分割传播推理（对应Python的predictor.propagate_in_video）
                            foreach (var result in predictor.PropagateInVideo(inferenceState))
                            {
                                // 解包推理结果（帧索引、目标ID、掩码预测结果）
                                var (outFrameIdx, resObjIds, resMaskLogits) = result;
                                // 无需额外处理，仅完成推理流程即可（性能测试关注整体耗时）
                            }

                            // 计时结束
                            stopwatch.Stop();
                            double elapsedSeconds = stopwatch.Elapsed.TotalSeconds;
                            totalElapsedSeconds += elapsedSeconds;
                            validRunCount++;

                            if (Verbose)
                                Console.WriteLine($" 完成，耗时：{elapsedSeconds:F4} 秒");

                            // 热身阶段结束，重置统计数据（仅保留正式测试结果）
                            if (i == WarmUp - 1)
                            {
                                double warmUpFps = (validRunCount * numFrames) / totalElapsedSeconds;
                                Console.WriteLine($"\n热身阶段完成，平均FPS：{warmUpFps:F2}（仅作参考，不参与正式统计）");
                                
                                // 重置统计值，开始正式测试
                                totalElapsedSeconds = 0.0;
                                validRunCount = 0;
                            }
                        }
                    }

                    // 7. 计算并输出正式测试结果
                    Console.WriteLine("\n=== 测试结果汇总 ===");
                    double averageFps = (validRunCount * numFrames) / totalElapsedSeconds;
                    Console.WriteLine($"正式测试运行次数：{validRunCount}");
                    Console.WriteLine($"单轮推理平均耗时：{totalElapsedSeconds / validRunCount:F4} 秒");
                    Console.WriteLine($"SAM2视频目标分割平均FPS：{averageFps:F2}");

                    // 8. 资源释放
                    inferenceState.Dispose();
                    predictor.Dispose();
                    try { Sam2Sharp.Utils.CudaHelpers.TryEmptyCache(); } catch { }
                    Console.WriteLine("\n=== 测试完成，资源已释放 ===");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n错误：测试过程中出现异常 - {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }

        #region 辅助方法（对应Python脚本的工具逻辑）
        /// <summary>
        /// 获取排序后的视频帧文件名（对应Python的文件遍历与排序）
        /// </summary>
        private static List<string> GetSortedFrameNames(string videoDir)
        {
            if (!Directory.Exists(videoDir))
                throw new DirectoryNotFoundException($"视频目录不存在：{videoDir}");

            // 支持的图片格式（与Python脚本一致）
            var validExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                ".jpg", ".jpeg", ".png"
            };

            // 遍历文件、筛选格式、按文件名数字排序
            return Directory.EnumerateFiles(videoDir)
                .Where(file => validExtensions.Contains(Path.GetExtension(file)))
                .OrderBy(file => 
                {
                    // 解析文件名中的数字（假设文件名格式为"xxx123.jpg"）
                    if (int.TryParse(Path.GetFileNameWithoutExtension(file), out int frameNum))
                        return frameNum;
                    return 0;
                })
                .ToList();
        }

        /// <summary>
        /// 构建标注点张量（对应Python的points构造）
        /// </summary>
        private static Tensor CreateAnnotationPoints()
        {
            // 构造形状：[1, 1, 2]（对应SAM2的输入格式：[批次, 点数量, 坐标维度]）
            return torch.tensor(new[,,] { { { AnnPoint.Item1, AnnPoint.Item2 } } }, 
                dtype: ScalarType.Float32, 
                device: torch.device(DeviceType.CUDA));
        }

        /// <summary>
        /// 构建SAM2视频预测器（占位实现，需对接实际SAM2 TorchSharp模型实现）
        /// </summary>
        private static Sam2VideoPredictor BuildSam2VideoPredictor(string configPath, string checkpointPath, Device device, bool vosOptimized)
        {
            // 注：此处为核心占位方法，实际需实现：
            // 1. 加载YAML配置文件（可使用YamlDotNet库）
            // 2. 从checkpoint加载模型权重（torch.load）
            // 3. 初始化SAM2VideoPredictor实例并配置参数
            // 此处简化返回自定义封装的预测器类，保持流程完整性
            if (!File.Exists(configPath))
                throw new FileNotFoundException("模型配置文件不存在", configPath);
            if (!File.Exists(checkpointPath))
                throw new FileNotFoundException("模型权重文件不存在", checkpointPath);

            return new Sam2VideoPredictor(configPath, checkpointPath, device, vosOptimized);
        }
        #endregion
    }

    #region SAM2核心类封装（适配TorchSharp，保持Python脚本接口语义）
    /// <summary>
    /// SAM2视频预测器封装类（对应Python的SAM2VideoPredictor）
    /// </summary>
    public class Sam2VideoPredictor : IDisposable
    {
        private readonly string _configPath;
        private readonly string _checkpointPath;
        private readonly Device _device;
        private readonly bool _vosOptimized;
        private bool _disposed = false;

        public Sam2VideoPredictor(string configPath, string checkpointPath, Device device, bool vosOptimized)
        {
            _configPath = configPath;
            _checkpointPath = checkpointPath;
            _device = device;
            _vosOptimized = vosOptimized;

            // 实际应在此处加载模型权重、初始化网络层等
        }

        /// <summary>
        /// 初始化推理状态（对应Python的init_state）
        /// </summary>
        public Sam2InferenceState InitState(string videoPath, bool offloadVideoToCpu, bool offloadStateToCpu, bool asyncLoadingFrames)
        {
            // 实际应在此处加载视频帧、初始化缓存、特征存储等
            return new Sam2InferenceState(videoPath, _device);
        }

        /// <summary>
        /// 添加新的标注点/边界框（对应Python的add_new_points_or_box）
        /// </summary>
        public (int, List<int>, Tensor) AddNewPointsOrBox(Sam2InferenceState inferenceState, int frameIdx, int objId, Tensor points, Tensor labels)
        {
            // 实际应在此处处理标注输入、运行单帧推理、更新推理状态
            var objIds = new List<int> { objId };
            var dummyMask = torch.zeros(1, 1, 256, 256, device: _device); // 占位掩码张量
            return (frameIdx, objIds, dummyMask);
        }

        /// <summary>
        /// 在视频中传播目标分割（对应Python的propagate_in_video）
        /// </summary>
        public IEnumerable<(int, List<int>, Tensor)> PropagateInVideo(Sam2InferenceState inferenceState)
        {
            // 实际应在此处实现时序传播逻辑、逐帧推理、返回掩码结果
            // 此处为占位实现，模拟逐帧返回结果
            var objIds = new List<int> { Program.AnnObjId };
            var dummyMask = torch.zeros(1, 1, 256, 256, device: _device);

            // 模拟视频帧数量（简化返回）
            int frameCount = Directory.EnumerateFiles(inferenceState.VideoPath).Count();
            for (int i = 0; i < frameCount; i++)
            {
                yield return (i, objIds, dummyMask);
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed) return;

            if (disposing)
            {
                // 释放托管资源（模型、张量等）
                Sam2Sharp.Utils.CudaHelpers.TryEmptyCache();
            }

            _disposed = true;
        }

        ~Sam2VideoPredictor()
        {
            Dispose(false);
        }
    }

    /// <summary>
    /// SAM2推理状态封装类（对应Python的推理状态对象）
    /// </summary>
    public class Sam2InferenceState : IDisposable
    {
        public string VideoPath ;
        public Device Device ;
        private bool _disposed = false;

        public Sam2InferenceState(string videoPath, Device device)
        {
            VideoPath = videoPath;
            Device = device;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed) return;

            if (disposing)
            {
                // 释放托管资源（缓存特征、掩码等）
                Sam2Sharp.Utils.CudaHelpers.TryEmptyCache();
            }

            _disposed = true;
        }

        ~Sam2InferenceState()
        {
            Dispose(false);
        }
    }
    #endregion
}