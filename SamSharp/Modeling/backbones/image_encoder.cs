using Sam2Sharp.Modeling.Backbones;
using Sam2Sharp.Modeling.PositionEncoding;
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static System.Runtime.InteropServices.JavaScript.JSType;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
namespace Sam2Sharp.Modeling.Backbones
{
    // ImageEncoder.cs
    public class ImageEncoder : Module<Tensor, Dictionary<string, object>>
    {
        private readonly Hiera trunk;
        public readonly FpnNeck neck;
        private readonly int scalp;
        public ImageEncoder(Hiera trunk, FpnNeck neck, int scalp = 0) : base("ImageEncoder")
        {
            this.trunk = trunk;
            this.neck = neck;
            this.scalp = scalp;
            // 验证通道匹配
            var trunkChannelList = trunk.channel_list;
            var neckChannelList = neck.backbone_channel_list;

            if (!trunkChannelList.SequenceEqual(neckChannelList))
                throw new ArgumentException($"通道维度不匹配. Trunk: {string.Join(",", trunkChannelList)}, Neck: {string.Join(",", neckChannelList)}");

            RegisterComponents();
        }

        public override Dictionary<string, object> forward(Tensor sample)
        {
            //using var _ = NewDisposeScope();
            var output = trunk.forward(sample);
            (List<Tensor>, List<Tensor>) oput = neck.forward(output);
            var features = oput.Item1;
            var pos = oput.Item2;
            if (scalp > 0)
            {
                features = features.Take(features.Count - scalp).ToList();
                pos = pos.Take(pos.Count - scalp).ToList();
            }

            var src = features.Last();
            var x = new Dictionary<string, object>
        {
            { "vision_features", src },
            { "vision_pos_enc", pos.ToArray() },
            { "backbone_fpn", features.ToArray() }
        };
            return x;
        }
        public  void Dispose(bool disposing)
        {
            if (disposing)
            {
                trunk?.Dispose();
                neck?.Dispose();
            }
            base.Dispose(disposing);
        }
    }

    // FpnNeck.cs
    public class FpnNeck : Module<List<Tensor>, (List<Tensor>,List<Tensor>)>
    {
        public int[] backbone_channel_list ;
        public int d_model ;
        private readonly PositionEmbeddingSine position_encoding;
        private readonly ModuleList<Module<Tensor,Tensor>> convs;
       // private readonly List<Sequential> convs;
        private readonly InterpolationMode fpn_interp_model = InterpolationMode.Bilinear;
        private readonly string fuse_type;
        private readonly int[] fpn_top_down_levels;
        public readonly ScalarType dtype;//上采样中有个Float32变换
        public FpnNeck(
            PositionEmbeddingSine position_encoding,
            int d_model,
            int[] backbone_channel_list,
            ScalarType dtype,
            int kernel_size = 1,
            int stride = 1,
            int padding = 0,
            InterpolationMode fpn_interp_model = InterpolationMode.Bilinear,
            string fuse_type = "sum",
            int[] fpn_top_down_levels = null) : base("FpnNeck")
        {
            this.position_encoding = position_encoding;
            this.d_model = d_model;
            this.backbone_channel_list = backbone_channel_list;
            this.dtype = dtype;
            this.convs = new ModuleList<Module<Tensor, Tensor>>();
            foreach (var dim in backbone_channel_list)
            {
                var current = Sequential(("conv",Conv2d(in_channels: dim,
                    out_channels: d_model,
                    kernel_size: kernel_size,
                    stride : stride,
                    padding : padding)));
                convs.Add(current);
            }
            this.fpn_interp_model = fpn_interp_model;
            this.fuse_type = fuse_type;
            if (fpn_top_down_levels is null)
                fpn_top_down_levels = Enumerable.Range(0, convs.Count).ToArray();
            this.fpn_top_down_levels = fpn_top_down_levels;
            RegisterComponents();
        }
        public override(List<Tensor>, List<Tensor>) forward(List<Tensor> xs)
        {
            int n = convs.Count;
            var outList = new List<Tensor>();
            var posList = new List<Tensor>();
            Tensor prev_features = null;

            for (int i = xs.Count - 1; i >= 0; i--)
            {
                var x = xs[i];
                var m = convs[xs.Count - 1 - i];
                var lateral_features = m.forward(x);
                if ((fpn_top_down_levels.Contains(i)) && (prev_features is not null))
                {
                    //var t = prev_features.flatten().reshape(1, prev_features.shape[2],;
                    var top_down_features = nn.functional.interpolate(
                        prev_features.to(float32),
                       scale_factor:[2.0,2.0],
                        mode: fpn_interp_model,
                        align_corners: fpn_interp_model == InterpolationMode.Nearest ? (bool?)null : false);
                    //var sparse_embeddings = torch.cat(new Tensor[] { lateral_features, top_down_features.to(this.dtype) }, dim: 1);
                    //prev_features = sparse_embeddings;
                    prev_features = lateral_features + top_down_features.to(this.dtype);
                    if (fuse_type == "avg")
                        prev_features /= 2;
                }
                else
                {
                    prev_features = lateral_features;
                }

                outList.Insert(0, prev_features);
                posList.Insert(0, position_encoding.forward(prev_features).to(prev_features.dtype));
            }

            return (outList, posList);
        }
        public void Dispose(bool disposing)
        {
            if (disposing)
            {
                position_encoding?.Dispose();
                foreach (var conv in convs)
                {
                    conv?.Dispose();
                }
            }
            base.Dispose(disposing);
        }
    }  
}
