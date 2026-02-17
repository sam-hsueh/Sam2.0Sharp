using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
namespace Sam2Sharp.Modeling
{
    //public class LayerNorm2d : Module<Tensor, Tensor>
    //{
    //    private readonly LayerNorm norm;

    //    public LayerNorm2d(long normalizedShape, float eps = 1e-5, float elementwiseAffine = 1)
    //        : base("LayerNorm2d")
    //    {
    //        norm = LayerNorm(new long[] { normalizedShape }, eps, elementwiseAffine);
    //        RegisterComponents();
    //    }

    //    public override Tensor forward(Tensor input)
    //    {
    //        // (N, C, H, W) -> (N, H, W, C) for layernorm
    //        var x = input.permute(0, 2, 3, 1);
    //        x = norm.forward(x);
    //        return x.permute(0, 3, 1, 2); // back to (N, C, H, W)
    //    }
    //}

    //public class drop_path : Module<Tensor, Tensor>
    //{
    //    private readonly float dropProb;
    //    private readonly bool training;

    //    public drop_path(float dropProb = 0.0) : base("drop_path")
    //    {
    //        this.dropProb = dropProb;
    //        RegisterComponents();
    //    }

    //    public override Tensor forward(Tensor input)
    //    {
    //        if (dropProb == 0.0 || !training)
    //            return input;

    //        var keepProb = 1 - dropProb;
    //        var shape = new long[] { input.size(0), 1, 1, 1 };
    //        var randomTensor = keepProb + rand(shape, device: input.device, dtype: input.dtype);
    //        var binaryTensor = floor(randomTensor);
    //        var output = input / keepProb * binaryTensor;
    //        return output;
    //    }
    //}


    public class MaskDownSampler : Module<Tensor, Tensor>
    {
        private readonly Sequential encoder;

        public MaskDownSampler(
            int embed_dim = 256,
            int kernel_size = 4,
            int stride = 4,
            int padding = 0,
            int total_stride = 16,
            Module<Tensor, Tensor> activation = null) : base("MaskDownSampler")
        {
            activation ??= GELU();
            var num_layers = (int)(Math.Log(total_stride, stride));
            if (Math.Pow(stride, num_layers) != total_stride)
                throw new ArgumentException("Invalid total stride and stride combination");

            encoder = Sequential();
            int maskInChans = 1, maskOutChans = 1;

            for (int i = 0; i < num_layers; i++)
            {
                maskOutChans = maskInChans * (stride * stride);
                encoder.append(Conv2d(maskInChans, maskOutChans, kernel_size, stride, padding));
                encoder.append(new LayerNorm2d(maskOutChans));
                encoder.append(activation);
                maskInChans = maskOutChans;
            }
            encoder.append(Conv2d(maskOutChans, embed_dim, 1));
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            return encoder.forward(x);
        }
    }

    public class CXBlock : Module<Tensor, Tensor>
    {
        private readonly Conv2d dwconv;
        private readonly LayerNorm2d norm;
        private readonly Linear pwconv1;
        private readonly Module<Tensor, Tensor> act;
        private readonly Linear pwconv2;
        private readonly Parameter gamma;
        private readonly Module<Tensor, Tensor> drop_path;
        public readonly int dim, kernel_size, padding;
        public readonly float layer_scale_init_value;
        public readonly bool use_dwconv;
        public readonly float drop_pathf;
        public CXBlock(
            int dim,
            int kernel_size = 7,
            int padding = 3,
            float drop_path = 0.0f,
            float layer_scale_init_value = 1e-6f,
            bool use_dwconv = true) : base("CXBlock")
        {
            this.dim = dim;
            this.kernel_size = kernel_size;
            this.padding = padding;
            this.layer_scale_init_value = layer_scale_init_value;
            this.drop_pathf = drop_path;
            this.use_dwconv = use_dwconv;

            dwconv = Conv2d(dim, dim, kernel_size, padding: padding, groups: use_dwconv ? dim : 1);
            norm = new LayerNorm2d(dim);
            pwconv1 = Linear(dim, 4 * dim);
            act = GELU();
            pwconv2 = Linear(4 * dim, dim);

            gamma = layer_scale_init_value > 0
                ? Parameter(torch.full(new long[] { dim }, layer_scale_init_value)) : null;

            this.drop_path = drop_path > 0 ? new DropPath(drop_path) : Identity();

            RegisterComponents();
            //if (gamma != null) RegisterParameter("gamma", gamma);
        }

        public override Tensor forward(Tensor x)
        {
            using var _ = NewDisposeScope();
            var input = x;
            x = dwconv.forward(x);
            x = norm.forward(x);
            x = x.permute(0, 2, 3, 1); // (N, C, H, W) -> (N, H, W, C)

            x = pwconv1.forward(x);
            x = act.forward(x);
            x = pwconv2.forward(x);

            if (gamma is not null)
                x = gamma * x;

            x = x.permute(0, 3, 1, 2); // (N, H, W, C) -> (N, C, H, W)
            x = input + drop_path.forward(x);

            return x;
        }
    }

    public class Fuser : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> proj;
        private readonly ModuleList<CXBlock> layers;

        public Fuser(CXBlock layer, int num_layers, int? dim = null, bool inputProjection = false)
            : base("Fuser")
        {
            layers = new ModuleList<CXBlock>();
            layers.Add(layer);
            for (int i = 0; i < num_layers-1; i++)
            {
                //var newLayer = new CXBlock(
                //    dim: layer.dim,
                //     kernel_size: layer.kernel_size,
                //     padding: layer.padding,
                //     layer_scale_init_value: layer.layer_scale_init_value,
                //     use_dwconv: layer.use_dwconv);
                layers.Add(layer);
            }
            proj = inputProjection && dim is not null
                ? Conv2d(dim!.Value, dim.Value, kernel_size: 1)
                : Identity();
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var _ = NewDisposeScope();
            x = proj.forward(x);
            foreach (var layer in layers)
                x = layer.forward(x);
            return x;
        }
    }

    public class MemoryEncoder : Module<Tensor, Tensor, bool, Dictionary<string, Tensor>>
    {
        public readonly MaskDownSampler mask_downsampler;
        private readonly Conv2d pix_feat_proj;
        private readonly Fuser fuser;
        private readonly Module<Tensor, Tensor> position_encoding;
        private readonly Module<Tensor, Tensor> out_proj;
        public readonly int out_dim;
        public readonly int in_dim;

        public MemoryEncoder(
            int out_dim,
            MaskDownSampler mask_downsampler,
            Fuser fuser,
            Module<Tensor, Tensor> position_encoding,
            int in_dim = 256) : base("MemoryEncoder")
        {
            this.in_dim = in_dim;
            this.out_dim = out_dim;
            this.mask_downsampler = mask_downsampler;
            this.pix_feat_proj = Conv2d(in_dim, in_dim, 1);
            this.fuser = fuser;
            this.position_encoding = position_encoding;
            this.out_proj = out_dim != in_dim ? Conv2d(in_dim, out_dim, 1) : Identity();            
            RegisterComponents();
        }

        public override Dictionary<string, Tensor> forward(Tensor pixFeat, Tensor masks, bool skip_mask_sigmoid=false)
        {
            using var _ = NewDisposeScope();
            // var (pixFeat, masks, skip_mask_sigmoid) = input;

            // Process masks
            var processedMasks = skip_mask_sigmoid ? masks : torch.sigmoid(masks);
            processedMasks = mask_downsampler.forward(processedMasks);

            // Fuse pix_feats and downsampled masks
            var pixFeatDevice = processedMasks.device;
            var x = pixFeat.to(pixFeatDevice);
            x = pix_feat_proj.forward(x);
            x = x + processedMasks;
            x = fuser.forward(x);
            x = out_proj.forward(x);

            var pos = position_encoding.forward(x).to(x.dtype);

            var result = new Dictionary<string, Tensor>
        {
            { "vision_features", x },
            { "backbone_fpn", pos.unsqueeze(0)}
        };
            return result;
            //return (
            //    new Dictionary<string, Tensor> { { "vision_features", x } },
            //    new Dictionary<string, Tensor> { { "vision_pos_enc", pos.unsqueeze(0) } }
            //);
        }
    }
    public class LayerNorm2d : Module<Tensor, Tensor>
    {
        private readonly Parameter weight;
        private readonly Parameter bias;
        private readonly float eps;

        public LayerNorm2d(int num_channels, float eps = 1e-6f) : base(nameof(LayerNorm2d))
        {
            this.weight = nn.Parameter(torch.ones(num_channels));
            this.bias = nn.Parameter(torch.zeros(num_channels));
            this.eps = eps;
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var _ = NewDisposeScope();
            Tensor u = x.mean(new long[] { 1 }, keepdim: true);
            Tensor s = (x - u).pow(2).mean(new long[] { 1 }, keepdim: true);
            x = (x - u) / torch.sqrt(s + this.eps);
            x = this.weight[.., TensorIndex.Null, TensorIndex.Null] * x + this.bias[.., TensorIndex.Null, TensorIndex.Null];
            return x.MoveToOuterDisposeScope();
        }

    }
}