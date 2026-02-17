using SAM2Sharp;
using System.IO;
using System.Threading;
using TorchSharp;
using TorchSharp.Modules;
using static Sam2Sharp.Modeling.Common;
using static Sam2Sharp.Modeling.Common.MLPBlock;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Sam2Sharp.Modeling.Backbones
{
    public static class HieraUtils
    {
        public static Tensor DoPool(Tensor x, Module<Tensor, Tensor> pool, Module<Tensor, Tensor> norm = null)
        {
            if (pool is null)
                return x;

            // (B, H, W, C) -> (B, C, H, W)
            x = x.permute(0, 3, 1, 2);
            x = pool.forward(x) as Tensor;

            // (B, C, H', W') -> (B, H', W', C)
            x = x.permute(0, 2, 3, 1);

            if (norm is not null)
                x = norm.forward(x) as Tensor;

            return x.MoveToOuterDisposeScope();
        }
    }

    public class MultiScaleAttention : Module<Tensor, Tensor>
    {
        public int dim;
        public int dim_out;
        public int num_heads;
        public Module<Tensor, Tensor> q_pool;
        public Linear qkv;
        public Linear proj;

        public MultiScaleAttention(int dim, int dim_out, int num_heads, Module<Tensor, Tensor> q_pool = null) : base("MultiScaleAttention")
        {
            this.dim = dim;
            this.dim_out = dim_out;
            this.num_heads = num_heads;
            this.q_pool = q_pool;

            this.qkv = Linear(dim, dim_out * 3);
            this.proj = Linear(dim_out, dim_out);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            long B = x.shape[0], H = x.shape[1], W = x.shape[2];
            // qkv with shape (B, H * W, 3, nHead, C)
            var QKV = qkv.forward(x).view(B, H * W, 3, num_heads, -1);
            // q, k, v with shape (B, H * W, nheads, C)
            var q = QKV.unbind(2)[0];
            var k = QKV.unbind(2)[1];
            var v = QKV.unbind(2)[2];

            // Q pooling (for downsample at stage changes)
            if (q_pool is not null)
            {
                q = q.view(B, H, W, -1);
                q = HieraUtils.DoPool(q, q_pool);
                H = q.shape[1];
                W = q.shape[2];
                q = q.view(B, H * W, num_heads, -1);
            }

            // Torch's SDPA expects [B, nheads, H*W, C] so we transpose
            x = functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2)
            );

            // Transpose back
            x = x.transpose(1, 2);
            x = x.view(B, H, W, -1);

            x = proj.forward(x) as Tensor;

            return x;
        }
        public void Dispose()
        {
            Dispose(true);
            q_pool.Dispose();
            qkv.Dispose();
            proj.Dispose();
            GC.SuppressFinalize(this);
        }
    }

    public class MultiScaleBlock : Module<Tensor, Tensor>
    {
        public int dim;
        public int dim_out;
        public Module<Tensor, Tensor> norm1;
        public readonly int window_size;
        public Module<Tensor, Tensor> pool;
        public int[] q_stride = [2, 2];
        public MultiScaleAttention attn;
        public Module<Tensor, Tensor> drop_path;
        public Module<Tensor, Tensor> norm2;
        public MLP mlp;
        internal Linear proj;

        public MultiScaleBlock(
            int dim,
            int dim_out,
            int num_heads,
            float mlp_ratio = 4.0f,
            float drop_path = 0.0f,
            Module<Tensor, Tensor> norm_layer = null,
            int[] q_stride = null,
            // Func<Module<Tensor,Tensor>> actLayer = nn.ReLU ,
            int window_size = 0) : base("MultiScaleBlock")
        {
            this.dim = dim;
            this.dim_out = dim_out;
            this.window_size = window_size;
            this.q_stride = q_stride;

            this.norm1 = norm_layer ?? LayerNorm(dim, eps: 1e-6);

            if (q_stride is not null && q_stride[0] > 0)
            {
                this.pool = MaxPool2d(kernel_size: q_stride[0], stride: q_stride[0], ceil_mode: false);
            }
            else
            {
                this.pool = null;
            }

            this.attn = new MultiScaleAttention(dim, dim_out, num_heads, pool);

            this.drop_path = drop_path > 0.0 ? new DropPath(drop_path) : Identity();

            this.norm2 = norm_layer ?? LayerNorm(dim_out, eps: 1e-6);

            this.mlp = new MLP(dim_out, (int)(dim_out * mlp_ratio), dim_out, 2, null);

            this.proj = dim != dim_out ? Linear(dim, dim_out) : null;

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            var shortcut = x; // B, H, W, C
            x = norm1.forward(x);

            // Skip connection
            if (dim != dim_out)
                shortcut = HieraUtils.DoPool(proj.forward(x), pool);

            // Window partition划分窗口
            Tensor xWindowed = null;
            (int, int) pad_hw = (0, 0);
            long H = 0, W = 0;

            var windowsize = this.window_size;
            if (windowsize > 0)
            {
                (H, W) = (x.shape[1], x.shape[2]);
                (x, pad_hw) = window_partition(x, window_size);
            }

            x = this.attn.forward(x);
            if (this.q_stride is not null && this.q_stride[0] > 0)
            {
                // Shapes have changed due to Q pooling
                windowsize = this.window_size / this.q_stride[0];
                (H,W) = (shortcut.shape[1], shortcut.shape[2]);
                var padH = (windowsize - H % windowsize) % windowsize;
                var padW = (windowsize - W % windowsize) % windowsize;
                pad_hw = ((int)(H + padH), (int)(W + padW));
            }
            if (this.window_size > 0)
                x = window_unpartition(x, windowsize, pad_hw, ((int)H, (int)W));

            x = shortcut + drop_path.forward(x);
            // MLP
            x = x + drop_path.forward(mlp.forward(norm2.forward(x))) as Tensor;
            return x;
        }
        public void Dispose()
        {
            Dispose(true);
            norm1.Dispose();
            pool.Dispose();
            attn.Dispose();
            drop_path.Dispose();
            norm2.Dispose();
            mlp.Dispose();
            proj.Dispose();
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Partition into non-overlapping windows with padding if needed.
        /// </summary>
        /// <param name="x">input tokens with [B, H, W, C].</param>
        /// <param name="window_size">window size.</param>
        /// <returns>windows:windows after partition with [B * num_windows, window_size, window_size, C].(Hp, Wp): padded height and width before partition</returns>
        private static (Tensor, (int, int)) window_partition(Tensor x, int window_size)
        {
            using var _ = NewDisposeScope();
            int B = (int)x.shape[0];
            int H = (int)x.shape[1];
            int W = (int)x.shape[2];
            int C = (int)x.shape[3];

            int pad_h = (window_size - H % window_size) % window_size;
            int pad_w = (window_size - W % window_size) % window_size;

            if (pad_h > 0 || pad_w > 0)
            {
                x = functional.pad(x, new long[] { 0, 0, 0, pad_w, 0, pad_h });
            }
            (int Hp, int Wp) = (H + pad_h, W + pad_w);
            x = x.view(B, Hp / window_size, window_size, Wp / window_size, window_size, C);

            Tensor windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C);
            return (windows.MoveToOuterDisposeScope(), (Hp, Wp));
        }

        /// <summary>
        /// Window unpartition into original sequences and removing padding.
        /// </summary>
        /// <param name="windows">input tokens with [B * num_windows, window_size, window_size, C].</param>
        /// <param name="window_size">window size.</param>
        /// <param name="pad_hw">padded height and width (Hp, Wp).</param>
        /// <param name="hw">original height and width (H, W) before padding.</param>
        /// <returns>unpartitioned sequences with [B, H, W, C].</returns>
        private static Tensor window_unpartition(Tensor windows, int window_size, (int, int) pad_hw, (int, int) hw)
        {
            using var _ = NewDisposeScope();
            (int Hp, int Wp) = pad_hw;
            (int H, int W) = hw;

            int B = (int)windows.shape[0] / (Hp * Wp / window_size / window_size);
            Tensor x = windows.view(B, Hp / window_size, Wp / window_size, window_size, window_size, -1);
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1);
            if (Hp > H || Wp > W)
            {
                x = x[.., ..H, ..W, ..].contiguous();
            }
            return x.MoveToOuterDisposeScope();

        }
    }

    public class Hiera : Module<Tensor, List<Tensor>>
    {
        int embed_dim = 96;  // initial embed dim
        int num_heads = 1;  // initial number of heads
        float drop_path_rate = 0.0f;  // stochastic depth
        int q_pool = 3;  // number of q_pool stages
        int[] q_stride = [2, 2];  // downsample stride bet. stages
        int[] stages = [2, 3, 16, 3];  // blocks per stage
        float dim_mul = 2.0f;  // dim_mul factor at stage shift
        float head_mul = 2.0f;  // head_mul factor at stage shift
        (int, int) window_pos_embed_bkg_spatial_size = (14, 14);
        // window size per stage, when not using global att.
        int[] window_spec = [8, 4, 14, 7];
        // global attn in these blocks
        int[] global_att_blocks = [12, 16, 20];
        string weights_path = "";
        bool return_interm_layers = true;  // return feats from every stage

        public PatchEmbed patch_embed;
        public Parameter pos_embed;
        public Parameter pos_embed_window;
        public ModuleList<MultiScaleBlock> blocks;
        public int[] stage_ends;
        public int[] q_pool_blocks;
        public int[] channel_list;

        //float drop_path_rate = 0.0f;

        public Hiera(
            int embed_dim = 96,
            int num_heads = 1,
            float drop_path_rate = 0.0f,
            int q_pool = 3,
            int[] q_stride = default,
            int[] stages = null,
            float dim_mul = 2.0f,
            float head_mul = 2.0f,
            int[] window_pos_embed_bkg_spatial_size = default,
            int[] window_spec = null,
            int[] global_att_blocks = null,
            string weights_path = null,
            bool return_interm_layers = true) : base("Hiera")
        {
            if (q_stride == default) q_stride = [2, 2];
            if (stages == null) stages = [2, 3, 16, 3];
            if (window_pos_embed_bkg_spatial_size == default) window_pos_embed_bkg_spatial_size = [14, 14];
            if (window_spec == null) window_spec = [8, 4, 14, 7];
            if (global_att_blocks == null) global_att_blocks = [12, 16, 20];

            this.embed_dim = embed_dim;
            this.num_heads = num_heads;
            this.drop_path_rate = drop_path_rate;
            this.q_pool = q_pool;
            this.q_stride = q_stride;
            this.dim_mul = dim_mul;
            this.head_mul = head_mul;
            this.window_spec = window_spec;
            this.return_interm_layers = return_interm_layers;

            if (stages.Length != window_spec.Length)
                throw new ArgumentException("Stages and window_spec must have the same length");

            var depth = stages.Sum();
            stage_ends = new int[stages.Length];
            var sum = 0;
            for (int i = 0; i < stages.Length; i++)
            {
                sum += stages[i];
                stage_ends[i] = sum - 1;
            }

            if (q_pool < 0 || q_pool > stage_ends.Length - 1)
                throw new ArgumentException("Invalid q_pool value");

            this.q_pool_blocks = stage_ends.Take(stage_ends.Length - 1).Take(q_pool).Select(x => x + 1).ToArray();

            this.patch_embed = new PatchEmbed(embed_dim: embed_dim);

            // Windowed positional embedding
            this.pos_embed = Parameter(torch.zeros(1, embed_dim, window_pos_embed_bkg_spatial_size[0], window_pos_embed_bkg_spatial_size[1]));
            this.pos_embed_window = Parameter(torch.zeros(1, embed_dim, window_spec[0], window_spec[0]));

            var dpr = Enumerable.Range(0, depth)
                .Select(i => torch.linspace(0, drop_path_rate, depth)[i].item<float>())
                .ToArray();

            var cur_stage = 1;
            this.blocks = ModuleList<MultiScaleBlock>();
            //var currentEmbed_dim = embed_dim;
            //var currentnum_heads = num_heads;

            for (int i = 0; i < depth; i++)
            {
                var dim_out = embed_dim;
                var window_size = window_spec[cur_stage - 1];

                if (global_att_blocks != null && global_att_blocks.Contains(i))
                    window_size = 0;

                //if (stage_ends.Contains(i - 1))
                if (stage_ends.Contains(i - 1))
                {
                    dim_out = (int)(embed_dim * dim_mul);
                    num_heads = (int)(num_heads * head_mul);
                    cur_stage++;
                }

                var q_strideVal = q_pool_blocks.Contains(i) ? q_stride : [0, 0];

                var block = new MultiScaleBlock(
                    dim: embed_dim,
                    dim_out: dim_out,
                    num_heads: num_heads,
                    drop_path: dpr[i],
                    q_stride: q_strideVal,
                    window_size: window_size
                );

                embed_dim = dim_out;
                this.blocks.Add(block);
            }

            this.channel_list = return_interm_layers
                ? stage_ends.Reverse().Select(i => blocks[i].dim_out).ToArray()
                : [ blocks.Last().dim_out ];

            if (!string.IsNullOrEmpty(weights_path))
            {
                // Implement weight loading logic
                // var chkpt = torch.load(weights_path);
                // LoadStateDict(chkpt);
                Console.WriteLine("Loading Hiera weights from " + weights_path);
            }

            RegisterComponents();
        }

        private Tensor get_pos_embed((long, long) hw)
        {
            long h = hw.Item1, w = hw.Item2;
            var window_embed = pos_embed_window;
            var window_embed_tensor = window_embed;
            var pos_embed_tensor = this.pos_embed;

            // 插值 + 计算tile次数 + 平铺 + 相加 一步完成
            var pos_embed_interp = functional.interpolate(pos_embed_tensor, new long[] { h, w },null, InterpolationMode.Bicubic, false);
            var pos_embed_final = pos_embed_interp + window_embed_tensor.repeat(
                Enumerable.Range(0, pos_embed_interp.shape.Length)
                          .Select(i => i < window_embed_tensor.shape.Length && window_embed_tensor.shape[i] != 0
                                      ? pos_embed_interp.shape[i] / window_embed_tensor.shape[i]
                                      : 1)
                          .ToArray()
            );
            return pos_embed_final.permute(0, 2, 3, 1);

            //var (h, w) = hw;
            //var windowEmbed = pos_embed_window;
            //var posEmbed = functional.interpolate(pos_embed, size: [h, w], mode: InterpolationMode.Bicubic);

            //// Calculate tiling factors
            //var tileH = posEmbed.shape[2] / windowEmbed.shape[2];
            //var tileW = posEmbed.shape[3] / windowEmbed.shape[3];

            //posEmbed = posEmbed + windowEmbed.repeat(1, 1, tileH, tileW);
            //posEmbed = posEmbed.permute(0, 2, 3, 1);

            //return posEmbed;
        }

        public override List<Tensor> forward(Tensor x)
        {
            x = patch_embed.forward(x) as Tensor;
            // x: (B, H, W, C)

            // Add pos embed
            x += get_pos_embed((x.shape[1], x.shape[2]));

            var outputs = new List<Tensor>();
            for (int i = 0; i < blocks.Count; i++)
            {
                x = blocks[i].forward(x) as Tensor;
                if (i == stage_ends.Last() || (stage_ends.Contains(i) && return_interm_layers))
                {
                    var feats = x.permute(0, 3, 1, 2);
                    outputs.Add(feats);
                }
            }
            return (outputs);
        }
        public void Dispose()
        {
            Dispose(true);
            patch_embed.Dispose();
            pos_embed.Dispose();
            pos_embed_window.Dispose();
            foreach (var block in blocks)
            {
                block.Dispose();
            }
            GC.SuppressFinalize(this);
        }

        public int get_layer_id(string layerName)
        {
            var num_layers = GetNumLayers();

            if (layerName.Contains("rel_pos"))
                return num_layers + 1;
            else if (layerName.Contains("pos_embed"))
                return 0;
            else if (layerName.Contains("patch_embed"))
                return 0;
            else if (layerName.Contains("blocks"))
            {
                var parts = layerName.Split(new[] { "blocks" }, StringSplitOptions.None);
                var blockNum = int.Parse(parts[1].Split('.')[1]);
                return blockNum + 1;
            }
            else
                return num_layers + 1;
        }

        public int GetNumLayers() => blocks.Count;
    }
}