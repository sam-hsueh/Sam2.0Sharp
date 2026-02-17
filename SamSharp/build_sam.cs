using System;
using System.Collections.Generic;
using System.IO;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAM2Sharp
{
    public static class SAM2Builder
    {
        private static readonly Dictionary<string, (string configPath, string checkpointName)> HfModelIdToFilenames = new()
        {
            {"facebook/sam2-hiera-tiny", ("configs/sam2/sam2_hiera_t.yaml", "sam2_hiera_tiny.pt")},
            {"facebook/sam2-hiera-small", ("configs/sam2/sam2_hiera_s.yaml", "sam2_hiera_small.pt")},
            {"facebook/sam2-hiera-base-plus", ("configs/sam2/sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt")},
            {"facebook/sam2-hiera-large", ("configs/sam2/sam2_hiera_l.yaml", "sam2_hiera_large.pt")},
            {"facebook/sam2.1-hiera-tiny", ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt")},
            {"facebook/sam2.1-hiera-small", ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt")},
            {"facebook/sam2.1-hiera-base-plus", ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt")},
            {"facebook/sam2.1-hiera-large", ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt")}
        };

        static SAM2Builder()
        {
            CheckPythonPathIssue();
        }

        private static void CheckPythonPathIssue()
        {
            var sam2Path = typeof(SAM2Builder).Assembly.Location;
            var parentDir = Path.GetDirectoryName(sam2Path);
            if (Directory.Exists(Path.Combine(parentDir, "sam2")))
            {
                throw new InvalidOperationException(
                    "检测到可能的路径冲突，请确保运行目录正确，避免SAM2库被同名目录遮挡。");
            }
        }

        public static Module BuildSam2(
            string configFile,
            string ckptPath = null,
            Device device = null,
            string mode = "eval",
            List<string> hydraOverridesExtra = null,
            bool applyPostprocessing = true,
            params (string key, object value)[] kwargs)
        {
            // Use torch.cuda.is_available to determine device
            try
            {
                device ??= torch.cuda.is_available() ? torch.device(DeviceType.CUDA) : torch.device(DeviceType.CPU);
            }
            catch
            {
                device ??= torch.device(DeviceType.CPU);
            }

            hydraOverridesExtra ??= new List<string>();

            if (applyPostprocessing)
            {
                hydraOverridesExtra.AddRange(new[]
                {
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98"
                });
            }

            var config = LoadConfig(configFile, hydraOverridesExtra);
            var model = InstantiateModel(config);

            if (!string.IsNullOrEmpty(ckptPath))
                LoadCheckpoint(model, ckptPath);

            // MoveTo / Eval may not be available; wrap in try/catch
            try { model.to(device); } catch { }
            try { model.eval(); } catch { }

            return model;
        }

        public static Module BuildSam2VideoPredictor(
            string configFile,
            string ckptPath = null,
            Device device = null,
            string mode = "eval",
            List<string> hydraOverridesExtra = null,
            bool applyPostprocessing = true,
            bool vosOptimized = false,
            params (string key, object value)[] kwargs)
        {
            try
            {
                device ??= torch.cuda.is_available() ? torch.device(DeviceType.CUDA) : torch.device(DeviceType.CPU);
            }
            catch
            {
                device ??= torch.device(DeviceType.CPU);
            }

            hydraOverridesExtra ??= new List<string>();

            var hydraOverrides = new List<string>();
            if (vosOptimized)
            {
                hydraOverrides.Add("++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS");
                hydraOverrides.Add("++model.compile_image_encoder=True");
            }
            else
            {
                hydraOverrides.Add("++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor");
            }

            if (applyPostprocessing)
            {
                hydraOverridesExtra.AddRange(new[]
                {
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",//Mask_decoder
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",//Mask_decoder
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",//Mask_decoder
                    "++model.binarize_mask_from_pts_for_mem_enc=true",//应用sam2_base
                    "++model.fill_hole_area=8"
                });
            }

            hydraOverrides.AddRange(hydraOverridesExtra);
            var config = LoadConfig(configFile, hydraOverrides);
            var model = InstantiateModel(config);

            if (!string.IsNullOrEmpty(ckptPath))
                LoadCheckpoint(model, ckptPath);

            try { model.to(device); } catch { }
            try { model.eval(); } catch { }

            return model;
        }

        public static Module BuildSam2Hf(string modelId, params (string key, object value)[] kwargs)
        {
            if (!HfModelIdToFilenames.TryGetValue(modelId, out var modelInfo))
                throw new KeyNotFoundException($"未找到模型ID: {modelId}");

            var (configName, ckptPath) = HfModelIdToFilenames[modelId];
            var localCkptPath = HfDownload(modelId, ckptPath);
            return BuildSam2(configName, localCkptPath, kwargs: kwargs);
        }

        public static Module BuildSam2VideoPredictorHf(string modelId, params (string key, object value)[] kwargs)
        {
            if (!HfModelIdToFilenames.TryGetValue(modelId, out var modelInfo))
                throw new KeyNotFoundException($"未找到模型ID: {modelId}");

            var (configName, ckptPath) = HfModelIdToFilenames[modelId];
            var localCkptPath = HfDownload(modelId, ckptPath);
            return BuildSam2VideoPredictor(configName, localCkptPath, kwargs: kwargs);
        }

        private static string HfDownload(string repoId, string filename)
        {
            var savePath = Path.Combine(Path.GetTempPath(), "sam2_cache", repoId, filename);
            Directory.CreateDirectory(Path.GetDirectoryName(savePath));
            return savePath;
        }

        private static void LoadCheckpoint(Module model, string ckptPath)
        {
            if (!File.Exists(ckptPath))
                throw new FileNotFoundException("检查点文件不存在", ckptPath);

            try
            {
                // 尝试将 state_dict 作为 Dictionary<string, torch.Tensor> 加载
                var stateDictObj = torch.load(ckptPath);
                if (stateDictObj is IDictionary<string, torch.Tensor> dict)
                {
                    try { model.load_state_dict((Dictionary<string, Tensor>)dict); } catch { }
                }
                else
                {
                    Console.WriteLine("加载的检查点不是 Dictionary<string, torch.Tensor> 类型，无法直接传递给 load_state_dict。");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"加载检查点失败: {ex.Message}");
            }

            Console.WriteLine("检查点加载尝试完成");
        }

        // 以下方法需要根据实际配置系统实现
        private static object LoadConfig(string configFile, List<string> overrides)
        {
            throw new NotImplementedException("配置加载逻辑需要实现");
        }

        private static Module InstantiateModel(object config)
        {
            throw new NotImplementedException("模型实例化逻辑需要实现");
        }
    }
}