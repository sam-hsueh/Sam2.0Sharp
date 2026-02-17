using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Sam2Sharp.Modeling
{
    public static class Sam2Utils
    {
            public static (Dictionary<int, T> selectedOutputs, Dictionary<int, T> unselectedOutputs)
            select_closest_cond_frames<T>(int frameIdx, Dictionary<int, T> condFrameOutputs, int maxCondFrameNum)
        {
            if (maxCondFrameNum == -1 || condFrameOutputs.Count <= maxCondFrameNum)
            {
                return (condFrameOutputs, new Dictionary<int, T>());
            }

            if (maxCondFrameNum < 2)
                throw new ArgumentException("max_cond_frame_num should be at least 2");

            var selectedOutputs = new Dictionary<int, T>();

            // Find closest before
            var beforeIndices = condFrameOutputs.Keys.Where(t => t < frameIdx).ToList();
            int? idxBefore = beforeIndices.Any() ? beforeIndices.Max() : (int?)null;
            if (idxBefore.HasValue)
                selectedOutputs[idxBefore.Value] = condFrameOutputs[idxBefore.Value];

            // Find closest after
            var afterIndices = condFrameOutputs.Keys.Where(t => t >= frameIdx).ToList();
            int? idxAfter = afterIndices.Any() ? afterIndices.Min() : (int?)null;
            if (idxAfter.HasValue)
                selectedOutputs[idxAfter.Value] = condFrameOutputs[idxAfter.Value];

            // Add remaining closest
            int numRemain = maxCondFrameNum - selectedOutputs.Count;
            var remainingIndices = condFrameOutputs.Keys
                .Where(t => !selectedOutputs.ContainsKey(t))
                .OrderBy(t => Math.Abs(t - frameIdx))
                .Take(numRemain)
                .ToList();

            foreach (var t in remainingIndices)
                selectedOutputs[t] = condFrameOutputs[t];

            var unselectedOutputs = condFrameOutputs
                .Where(kv => !selectedOutputs.ContainsKey(kv.Key))
                .ToDictionary(kv => kv.Key, kv => kv.Value);

            return (selectedOutputs, unselectedOutputs);
        }

        public static Tensor get_1d_sine_pe(Tensor posInds, int dim, float temperature = 10000f)
        {
            int peDim = dim / 2;
            var dimT = arange(peDim, dtype: ScalarType.Float32, device: posInds.device)
                .pow(2 * (arange(peDim, dtype: ScalarType.Float32, device: posInds.device) / 2) / peDim)
                .mul(torch.log(torch.tensor(temperature))).exp();

            var posEmbed = posInds.unsqueeze(-1).div(dimT);
            return torch.cat(new[] { posEmbed.sin(), posEmbed.cos() }, dim: -1);
        }

        public static Func<Tensor,bool, Tensor> GetActivationFn(string activation)
        {
            return activation switch
            {
                "relu" => functional.relu,
                "gelu" => functional.gelu,
                "glu" => functional.gelu,
                _ => throw new ArgumentException($"activation should be relu/gelu, not {activation}.")
            };
        }

        //public static ModuleList<Module> GetClones(Module module, int n)
        //{
        //    var clones = new ModuleList<Module>();
        //    for (int i = 0; i < n; i++)
        //        clones.Add(module); // reuse same module instance for compile-time compatibility
        //    return clones;
        //}


        public static (Tensor boxCoords, Tensor boxLabels) sample_box_points(
            Tensor masks, float noise = 0.1f, int noiseBound = 20,
            int topLeftLabel = 2, int bottomRightLabel = 3)
        {
            var device = masks.device;
            var boxCoords = MaskToBox(masks); // 需要实现mask_to_box函数
            var shape = masks.shape;
            long B = shape[0], H = shape[2], W = shape[3];

            var boxLabels = torch.tensor(new[] { topLeftLabel, bottomRightLabel }, dtype: ScalarType.Int32, device: device)
                .repeat(B)
                .reshape(B, 2);

            if (noise > 0.0)
            {
                var noiseBoundTensor = torch.tensor(noiseBound, device: device);
                var bboxW = boxCoords[.., 2] - boxCoords[.., 0];
                var bboxH = boxCoords[.., 3] - boxCoords[.., 1];
                var maxDx = torch.min(bboxW * noise, noiseBoundTensor);
                var maxDy = torch.min(bboxH * noise, noiseBoundTensor);

                var boxNoise = 2 * torch.rand(B, 1, 4, device: device) - 1;
                boxNoise = boxNoise * torch.stack(new[] { maxDx, maxDy, maxDx, maxDy }, dim: -1);

                boxCoords = boxCoords + boxNoise;
                var imgBounds = torch.tensor(new[] { W - 1, H - 1, W - 1, H - 1 }, device: device);
                boxCoords = torch.clamp(boxCoords, torch.zeros_like(imgBounds), imgBounds);
            }

            boxCoords = boxCoords.reshape(-1, 2, 2);
            return (boxCoords, boxLabels);
        }

        public static (Tensor points, Tensor labels) SampleRandomPointsFromErrors(
            Tensor gtMasks, Tensor predMasks, int numPt = 1)
        {
            if (predMasks is null)
                predMasks = torch.zeros_like(gtMasks, dtype: ScalarType.Bool);

            if (gtMasks.dtype != ScalarType.Bool || gtMasks.size(1) != 1)
                throw new ArgumentException("gtMasks must be boolean with size 1 in dimension 1");

            if (predMasks.dtype != ScalarType.Bool || !predMasks.shape.SequenceEqual(gtMasks.shape))
                throw new ArgumentException("predMasks must be boolean with same shape as gtMasks");

            var shape = gtMasks.shape;
            long B = shape[0], H_im = shape[2], W_im = shape[3];
            var device = gtMasks.device;

            // False positive and negative masks
            var fpMasks = ~gtMasks & predMasks;
            var fnMasks = gtMasks & ~predMasks;
            var allCorrect = torch.all((gtMasks == predMasks).flatten(2), dim: 2).unsqueeze(-1).unsqueeze(-1);

            // Sample points
            var ptsNoise = torch.rand(B, numPt, H_im, W_im,ScalarType.Float32, device: device);
            ptsNoise[.., .., .., .., 0] *= fpMasks | (allCorrect & ~gtMasks);
            ptsNoise[.., .., .., .., 1] *= fnMasks;

            var ptsIdx = ptsNoise.flatten(2).argmax(dim: 2);
            var labels = (ptsIdx % 2).to(ScalarType.Int32);
            ptsIdx = ptsIdx / 2;

            var ptsX = ptsIdx % W_im;
            var ptsY = ptsIdx / W_im;
            var points = torch.stack(new[] { ptsX, ptsY }, dim: 2).to(ScalarType.Float32);

            return (points, labels);
        }

        // 注意：SampleOnePointFromErrorCenter需要OpenCV的C#绑定
        // 这里仅提供框架，实际实现需要添加OpenCV相关代码
        public static (Tensor points, Tensor labels) SampleOnePointFromErrorCenter(
            Tensor gtMasks, Tensor predMasks, bool padding = true)
        {
            // 实现需要OpenCVSharp等库支持
            throw new NotImplementedException("SampleOnePointFromErrorCenter requires OpenCV bindings");
        }

        public static (Tensor points, Tensor labels) GetNextPoint(
            Tensor gtMasks, Tensor predMasks, string method)
        {
            return method switch
            {
                "uniform" => SampleRandomPointsFromErrors(gtMasks, predMasks),
                "center" => SampleOnePointFromErrorCenter(gtMasks, predMasks),
                _ => throw new ArgumentException($"Unknown sampling method {method}")
            };
        }

        /// <summary>
        /// 从二值掩码张量中提取每个掩码的边界框坐标。
        /// 输入: masks [B, 1, H, W]，输出: [B, 4]，每行为[x_min, y_min, x_max, y_max]
        /// </summary>
        /// <param name="masks">掩码张量，形状为[B, 1, H, W]，类型为Bool</param>
        /// <returns>边界框坐标张量，形状为[B, 4]</returns>
        public static Tensor MaskToBox(Tensor masks)
        {
            if (masks.dim() != 4 || masks.size(1) != 1)
                throw new ArgumentException("masks must be of shape [B, 1, H, W]");

            var device = masks.device;
            var dtype = ScalarType.Float32;
            long B = masks.size(0);
            long H = masks.size(2);
            long W = masks.size(3);

            // 计算每个掩码的非零像素坐标
            var boxes = torch.zeros(B, 4, dtype: dtype, device: device);
            for (long b = 0; b < B; b++)
            {
                var mask = masks[b, 0]; // [H, W]
                var nonzero = mask.nonzero();
                if (nonzero.shape[0] == 0)
                {
                    // 如果没有前景，返回全0
                    boxes[b] = torch.zeros(4, dtype: dtype, device: device);
                    continue;
                }
                var y = nonzero[.., 0];
                var x = nonzero[.., 1];
                var xMin = x.min().to_type(dtype);
                var yMin = y.min().to_type(dtype);
                var xMax = x.max().to_type(dtype);
                var yMax = y.max().to_type(dtype);
                boxes[b, 0] = xMin;
                boxes[b, 1] = yMin;
                boxes[b, 2] = xMax;
                boxes[b, 3] = yMax;
            }
            return boxes;
        }
    }
    public class DropPath : Module<Tensor, Tensor>
    {
        private double drop_prob;
        private bool scale_by_keep;

        public DropPath(double dropProb = 0.0, bool scaleByKeep = true) : base("DropPath")
        {
            drop_prob = dropProb;
            scale_by_keep = scaleByKeep;
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using (NewDisposeScope())
            {
                if (drop_prob == 0.0 || !training)
                {
                    return x;
                }

                var keep_prob = 1 - drop_prob;
                var shape = new long[] { x.shape[0] }.Concat(Enumerable.Repeat(1L, (int)x.ndim - 1)).ToArray();
                var random_tensor = x.new_empty(shape).bernoulli_(keep_prob);

                if (keep_prob > 0.0 && scale_by_keep)
                {
                    random_tensor.div_(keep_prob);
                }
                return (x * random_tensor).MoveToOuterDisposeScope();
            }
        }
    }

    public class MLP : Module<Tensor, Tensor>
    {
        private readonly int _num_layers;
        private readonly ModuleList<Linear> layers;
        private readonly bool sigmoid_output;
        private readonly Module<Tensor, Tensor> act = GELU();
        public MLP(int input_dim, int hidden_dim, int output_dim, int num_layers, Module<Tensor, Tensor> activation = null, bool sigmoid_output = false) : base("MLP")
        {
            _num_layers = num_layers;
            var layerSizes = new List<int> { input_dim };
            layerSizes.AddRange(Enumerable.Repeat(hidden_dim, num_layers - 1));
            layerSizes.Add(output_dim);

            layers = new ModuleList<Linear>();
            for (int i = 0; i < num_layers; i++)
            {
                layers.Add(Linear(layerSizes[i], layerSizes[i + 1]));
            }
            this.sigmoid_output = sigmoid_output;
            if (activation is not null)
                act = activation;
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                x = ((Linear)layers[i]).forward(x);
                if (i < layers.Count - 1)
                    x = act.forward(x);
            }

            if (sigmoid_output)
                x = sigmoid(x);

            return x;
        }
        public void Dispose()
        {
            Dispose(true);
            foreach (var layer in layers)
            {
                layer.Dispose();
            }
            GC.SuppressFinalize(this);
        }
    }

    //public class LayerNorm2d : Module
    //{
    //    private readonly Parameter _weight;
    //    private readonly Parameter _bias;
    //    private readonly double _eps;

    //    public LayerNorm2d(int numChannels, double eps = 1e-6) : base("LayerNorm2d")
    //    {
    //        _weight = Parameter(torch.ones(numChannels));
    //        _bias = Parameter(torch.zeros(numChannels));
    //        _eps = eps;
    //        RegisterComponents();
    //    }

    //    public override Tensor forward(Tensor x)
    //    {
    //        var u = x.mean(1, keepdim: true);
    //        var s = (x - u).pow(2).mean(1, keepdim: true);
    //        x = (x - u) / torch.sqrt(s + _eps);
    //        x = _weight.unsqueeze(1).unsqueeze(1) * x + _bias.unsqueeze(1).unsqueeze(1);
    //        return x;
    //    }
    //}
}