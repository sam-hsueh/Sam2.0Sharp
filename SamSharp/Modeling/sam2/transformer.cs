using Sam2Sharp.Modeling.PositionEncoding;
using SAM2Sharp;
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static Sam2Sharp.Modeling.Common;
using static TorchSharp.Modules.TransformerEncoder;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
namespace Sam2Sharp.Modeling.Sam2
{
    public class Transformer
    {
        public class TwoWayTransformer : Module<Tensor, Tensor, Tensor, (Tensor, Tensor)>
        {
            private readonly int depth;
            private readonly int embedding_dim;
            private readonly int num_heads;
            private readonly int mlp_dim;
            private readonly ModuleList<TwoWayAttentionBlock> layers;
            private readonly Attention final_attn_token_to_image;
            private readonly LayerNorm norm_final_attn;
            private readonly Module<Tensor, Tensor> act = nn.ReLU();
            int attention_downsample_rate = 2;
            public TwoWayTransformer(
                int depth,
                int embedding_dim,
                int num_heads,
                int mlp_dim,
                Module<Tensor, Tensor> activation = null,
                int attention_downsample_rate = 2) : base("TwoWayTransformer")
            {

                //    activation = activation?? ReLU();

                this.depth = depth;
                this.embedding_dim = embedding_dim;
                this.num_heads = num_heads;
                this.mlp_dim = mlp_dim;
                this.layers = new ModuleList<TwoWayAttentionBlock>();
                for (int i = 0; i < depth; i++)
                {
                    this.layers.append(new TwoWayAttentionBlock(
                    embedding_dim: embedding_dim,
                    num_heads: num_heads,
                    mlp_dim: mlp_dim,
                    activation: activation,
                    attention_downsample_rate: attention_downsample_rate,
                    skip_first_layer_pe: (i == 0)));
                }
                this.final_attn_token_to_image = new Attention(embedding_dim, num_heads, downsample_rate: attention_downsample_rate);
                this.norm_final_attn = nn.LayerNorm(embedding_dim);
                RegisterComponents();
            }

            public override (Tensor, Tensor) forward(Tensor image_embedding, Tensor image_pe, Tensor point_embedding)
            {
                //// BxCxHxW -> BxHWxC == B x N_image_tokens x C
                //var (bs, c, h, w) = (image_embedding.shape[0], image_embedding.shape[1], 
                //                    image_embedding.shape[2], image_embedding.shape[3]);

                //var imageEmbedding = image_embedding.flatten(2).permute(0, 2, 1);
                //var imagePe = image_pe.flatten(2).permute(0, 2, 1);

                //// Prepare queries and keys
                //var queries = point_embedding;
                //var keys = imageEmbedding;

                //// Apply transformer blocks
                //foreach (var layer in layers)
                //{
                //    (queries, keys) = layer.forward(queries, keys, point_embedding, imagePe);
                //}

                //// Apply final attention layer
                //var q = queries + point_embedding;
                //var k = keys + imagePe;
                //var attnOut = final_attn_token_to_image.forward(q, k, keys);
                //queries = queries + attnOut;
                //queries = norm_final_attn.forward(queries);

                //return (queries, keys);
                //using var _ = NewDisposeScope();
                // BxCxHxW -> BxHWxC == B X N_image_tokens X C
                long bs = image_embedding.shape[0];
                long c = image_embedding.shape[1];
                long h = image_embedding.shape[2];
                long w = image_embedding.shape[3];

                image_embedding = image_embedding.flatten(2).permute(0, 2, 1);
                image_pe = image_pe.flatten(2).permute(0, 2, 1);

                // Prepare queries
                Tensor queries = point_embedding;
                Tensor keys = image_embedding;

                // Apply Transformer blocks and final layernorm
                foreach (var layer in this.layers)
                {
                    (queries, keys) = layer.forward(queries: queries, keys: keys, query_pe: point_embedding, key_pe: image_pe);
                }
                // Apply the final attention layer from the points to the image
                Tensor q = queries + point_embedding;
                Tensor k = keys + image_pe;
                Tensor attn_out = this.final_attn_token_to_image.forward(q: q, k: k, v: keys);
                queries = queries + attn_out;
                queries = this.norm_final_attn.forward(queries);
                return (queries.MoveToOuterDisposeScope(), keys.MoveToOuterDisposeScope());
            }
        }

        public class TwoWayAttentionBlock : Module<Tensor, Tensor, Tensor, Tensor, (Tensor, Tensor)>
        {
            private readonly Attention self_attn;
            private readonly LayerNorm norm1;
            private readonly Attention cross_attn_token_to_image;
            private readonly LayerNorm norm2;
            private readonly MLP mlp;
            private readonly LayerNorm norm3;
            private readonly LayerNorm norm4;
            private readonly Attention cross_attn_image_to_token;
            private readonly bool skip_first_layer_pe;
            public TwoWayAttentionBlock(
                int embedding_dim,
                int num_heads,
                int mlp_dim = 2048,
                Module<Tensor, Tensor> activation = null,
                int attention_downsample_rate = 2,
                bool skip_first_layer_pe = false) : base("TwoWayAttentionBlock")
            {
                this.self_attn = new Attention(embedding_dim, num_heads);
                this.norm1 = nn.LayerNorm(embedding_dim);
                this.cross_attn_token_to_image = new Attention(embedding_dim, num_heads, downsample_rate: attention_downsample_rate);
                this.norm2 = nn.LayerNorm(embedding_dim);
                this.mlp = new MLP(embedding_dim, mlp_dim, embedding_dim, num_layers:2, activation:activation);
                this.norm3 = nn.LayerNorm(embedding_dim);
                this.norm4 = nn.LayerNorm(embedding_dim);
                this.cross_attn_image_to_token = new Attention(embedding_dim, num_heads, downsample_rate: attention_downsample_rate);
                this.skip_first_layer_pe = skip_first_layer_pe;
                RegisterComponents();

                //self_attn = new Attention(embedding_dim, num_heads);
                //norm1 = LayerNorm(embedding_dim);

                //cross_attn_token_to_image = new Attention(embedding_dim, num_heads, attention_downsample_rate);
                //norm2 = LayerNorm(embedding_dim);

                //activation ??= () => function.relu;
                //mlp = new MLP(embedding_dim, mlp_dim, embedding_dim, 2, activation);
                //norm3 = LayerNorm(embedding_dim);

                //norm4 = LayerNorm(embedding_dim);
                //cross_attn_to_token = new Attention(embedding_dim, num_heads, attention_downsample_rate);

                //this.skip_first_layer_pe = skip_first_layer_pe;

                //RegisterComponents();
            }

            public override (Tensor, Tensor) forward(Tensor queries, Tensor keys, Tensor query_pe, Tensor key_pe)
            {
                //// Self attention block
                //Tensor q = torch.zeros(0);
                //Tensor attn_out = torch.zeros(0);
                //if (skip_first_layer_pe)
                //{
                //    queries = self_attn.forward(queries, queries, queries);
                //}
                //else
                //{
                //    var q0 = queries + query_pe;
                //    var attnOut = self_attn.forward(q0, q0, queries);
                //    queries = queries + attnOut;
                //}
                //queries = norm1.forward(queries);

                //// Cross attention block, tokens attending to image embedding
                //var q1 = queries + query_pe;
                //var k1 = keys + key_pe;
                //var attnOut1 = cross_attn_token_to_image.forward(q1, k1, keys);
                //queries = queries + attnOut1;
                //queries = norm2.forward(queries);

                //// MLP block
                //var mlpOut = mlp.forward(queries);
                //queries = queries + mlpOut;
                //queries = norm3.forward(queries);

                //// Cross attention block, image embedding attending to tokens
                //var q2 = queries + query_pe;
                //var k2 = keys + key_pe;
                //var attnOut2 = cross_attn_to_token.forward(k2, q2, queries);
                //keys = keys + attnOut2;
                //keys = norm4.forward(keys);

                //return (queries.MoveToOuterDisposeScope(), keys.MoveToOuterDisposeScope());

                Tensor q = torch.zeros(0);
                Tensor attn_out = torch.zeros(0);
                if (this.skip_first_layer_pe)
                {
                    queries = this.self_attn.forward(q: queries, k: queries, v: queries);
                }
                else
                {
                    q = queries + query_pe;
                    attn_out = this.self_attn.forward(q: q, k: q, v: queries);
                    queries = queries + attn_out;
                }
                queries = this.norm1.forward(queries);

                // Cross attention block, tokens attending to image embedding
                q = queries + query_pe;
                Tensor k = keys + key_pe;
                attn_out = this.cross_attn_token_to_image.forward(q: q, k: k, v: keys);
                queries = queries + attn_out;
                queries = this.norm2.forward(queries);

                // MLP block
                Tensor mlp_out = this.mlp.forward(queries);
                queries = queries + mlp_out;
                queries = this.norm3.forward(queries);

                // Cross attention block, image embedding attending to tokens
                q = queries + query_pe;
                k = keys + key_pe;
                attn_out = this.cross_attn_image_to_token.forward(q: k, k: q, v: queries);
                keys = keys + attn_out;
                keys = this.norm4.forward(keys);
                return (queries.MoveToOuterDisposeScope(), keys.MoveToOuterDisposeScope());
            }
        }

        public class Attention : Module<Tensor, Tensor, Tensor, int, Tensor>
        {
            public readonly int embedding_dim;
            public readonly int internal_dim;
            public readonly int kv_in_dim;
            public readonly int num_heads;
            public Linear q_proj;
            public Linear k_proj;
            public Linear v_proj;
            public Linear out_proj;
            public float dropout_p = 0;

            public Attention(int embedding_dim, int num_heads, int downsample_rate = 1, float dropout = 0f, int kv_in_dim = 0) : base(nameof(Attention))
            {
                this.embedding_dim = embedding_dim;
                this.internal_dim = embedding_dim / downsample_rate;
                this.num_heads = num_heads;
                this.dropout_p = dropout;
                this.kv_in_dim = kv_in_dim == 0 ? embedding_dim : kv_in_dim;
                if (this.internal_dim % num_heads != 0)
                {
                    throw new ArgumentException("num_heads must divide embedding_dim.", nameof(num_heads));
                }
                this.q_proj = nn.Linear(embedding_dim, this.internal_dim);
                this.k_proj = nn.Linear(this.kv_in_dim, this.internal_dim);
                this.v_proj = nn.Linear(this.kv_in_dim, this.internal_dim);
                this.out_proj = nn.Linear(this.internal_dim, embedding_dim);

                //   this.q_proj = nn.Linear(embedding_dim, this.internal_dim);
                ////   internal_dim = kv_in_dim == 0 ? internal_dim : kv_in_dim;
                //   this.k_proj = nn.Linear(embedding_dim, this.internal_dim);
                //   this.v_proj = nn.Linear(embedding_dim, this.internal_dim);
                //   this.out_proj = nn.Linear(this.internal_dim, embedding_dim);
                //   if (kv_in_dim != 0)
                //   {
                //       this.k_proj = nn.Linear(kv_in_dim, this.internal_dim);
                //       this.v_proj = nn.Linear(kv_in_dim, this.internal_dim);
                //   }

                RegisterComponents();
            }
            public Tensor _separate_heads(Tensor x, int num_heads)
            {
                long b = x.shape[0];
                long n = x.shape[1];
                long c = x.shape[2];
                x = x.reshape(b, n, num_heads, c / num_heads);
                return x.transpose(1, 2);  // B X N_heads X N_tokens X C_per_head
            }

            public Tensor _recombine_heads(Tensor x)
            {
                long b = x.shape[0];
                long n_heads = x.shape[1];
                long n_tokens = x.shape[2];
                long c_per_head = x.shape[3];
                x = x.transpose(1, 2);  // B X N_tokens X N_heads X C_per_head
                return x.reshape(b, n_tokens, n_heads * c_per_head);  // B X N_tokens X C
            }

            public override Tensor forward(Tensor q, Tensor k, Tensor v, int kw = 0)
            {
                using var _ = NewDisposeScope();
                q = this.q_proj.forward(q);
                k = this.k_proj.forward(k);
                v = this.v_proj.forward(v);

                // Separate into heads
                q = this._separate_heads(q, this.num_heads);
                k = this._separate_heads(k, this.num_heads);
                v = this._separate_heads(v, this.num_heads);

                // Attention
                long c_per_head = q.shape[3];

                Tensor attn = q.matmul(k.permute(0, 1, 3, 2));  // B X N_heads X N_tokens X N_tokens;
                attn = attn / Math.Sqrt(c_per_head);
                attn = torch.softmax(attn, dim: -1);

                // Get output
                Tensor @out = attn.matmul(v);
                @out = this._recombine_heads(@out);
                @out = this.out_proj.forward(@out);
                return @out.MoveToOuterDisposeScope();
            }
        }
        public class RoPEAttention : Attention
        {
            private readonly Func<int, int, Tensor> compute_cis;
            private Tensor freqs_cis;
            public readonly bool rope_k_repeat;
            public readonly int[] feat_sizes;
            public readonly int embedding_dim;
            public readonly int num_heads;
            public readonly int downsample_rate;
            public readonly float dropout = 0.0f;
            public readonly int kv_in_dim = 0;
            public readonly float rope_theta = 10000;
            public RoPEAttention(int embedding_dim, int num_heads, int downsample_rate = 1, float dropout = 0.0f, int kv_in_dim = 0, float rope_theta = 10000, bool rope_k_repeat = false, int[] feat_sizes = default) : base(embedding_dim, num_heads, downsample_rate, dropout, kv_in_dim)
            {
                if (feat_sizes == default)
                    feat_sizes = [64, 64]; // Default for stride 16 feats at 1024 resolution
                this.feat_sizes = feat_sizes;
                this.rope_k_repeat = rope_k_repeat;
                this.embedding_dim= embedding_dim;
            this.num_heads= num_heads;
            this.downsample_rate = downsample_rate;
            this.dropout = dropout;
            this.kv_in_dim = kv_in_dim;
            this.rope_theta = rope_theta;
            compute_cis = (end_x, end_y) => RotaryPositionalEncoding.compute_axial_cis(internal_dim / num_heads, end_x, end_y, rope_theta);
                freqs_cis = compute_cis(feat_sizes[0], feat_sizes[1]);
                if (torch.cuda.is_available())
                    freqs_cis = freqs_cis.cuda();
                this.rope_k_repeat = rope_k_repeat;
            }

            public Tensor forward(Tensor q, Tensor k, Tensor v, int num_k_exclude_rope = 0)
            {
                using var _ = NewDisposeScope();
                // Input projections
                q = q_proj.forward(q);
                k = k_proj.forward(k);
                v = v_proj.forward(v);

                // Separate into heads
                q = _separate_heads(q, num_heads);
                k = _separate_heads(k, num_heads);
                v = _separate_heads(v, num_heads);

                // Apply rotary position encoding
                var w = Math.Sqrt(q.shape[2]);
                var h = Math.Sqrt(q.shape[2]);

                freqs_cis = freqs_cis.to(q.device);
                if (freqs_cis.shape[0] != q.shape[2])
                    freqs_cis = compute_cis((int)w, (int)h).to(q.device);

                if (q.shape[2] != k.shape[2] && !rope_k_repeat)
                    throw new InvalidOperationException("q and k have different lengths but rope_k_repeat is false");

                var num_k_rope = k.size(2) - num_k_exclude_rope;
                var (qRot, kRot) = RotaryPositionalEncoding.apply_rotary_enc(q, k[.., .., num_k_rope, ..], freqs_cis, rope_k_repeat);

                q = qRot;
                k[.., .., num_k_rope, ..] = kRot;

                var dropoutP = training ? dropout_p : 0.0;
                // Attention
                var output = functional.scaled_dot_product_attention(q, k, v, dropoutP);

                output = _recombine_heads(output);
                output = out_proj.forward(output);

                return output;
            }
            public void Dispose(bool disposing)
            {
                if (disposing)
                {
                    freqs_cis?.Dispose();
                }
                base.Dispose(disposing);
            }
        }
    }
    class MLP : Module<Tensor, Tensor>
    {
        int input_dim;
        int hidden_dim;
        int output_dim;
        int num_layers;
        Module<Tensor, Tensor> act = nn.ReLU();
        bool sigmoid_output = false;
        private readonly ModuleList<Linear> layers;

        public MLP(int input_dim,  int hidden_dim,  int output_dim,   int num_layers,  Module<Tensor, Tensor> activation=null, bool sigmoid_output = false) : base(nameof(MLP))
        {
            this.num_layers = num_layers;
            // 1. 构造隐藏层维度列表（对应Python的h = [hidden_dim] * (num_layers - 1)）
            // 修正原Python笔误：数值相乘→列表重复
            var h = Enumerable.Repeat(hidden_dim, num_layers - 1).ToList();

            // 2. 构造输入维度列表：[input_dim] + h → C#用Concat拼接
            var inputDims = new List<int> { input_dim }.Concat(h).ToList();

            // 3. 构造输出维度列表：h + [output_dim]
            var outputDims = h.Concat(new List<int> { output_dim }).ToList();

            // 4. 遍历维度对生成Linear层（对应Python的zip + 生成器）
            layers = new ModuleList<Linear>();
            this.sigmoid_output = sigmoid_output;

            this.act = activation??nn.ReLU();
            for (int i = 0; i < inputDims.Count; i++)
            {
                int n = inputDims[i];  // 当前层输入维度
                int k = outputDims[i]; // 当前层输出维度
                                       // 创建Linear层（TorchSharp的nn.Linear）
                var linear = nn.Linear(n, k);
                layers.Add(linear);
            }

            // 5. 封装为ModuleList（对应Python的nn.ModuleList）
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
           // using var _ = NewDisposeScope();
            // 核心前向传播逻辑
            for (int i = 0; i < layers.Count; i++)
            {
                var layer = layers[i];
                var layerOutput = layer.forward(x);

                if (i < this.num_layers - 1)
                {
                    x = this.act.forward(layerOutput);
                }
                else
                {
                    x = layerOutput;
                }
            }

            // 可选Sigmoid输出
            if (this.sigmoid_output)
            {
                x = functional.sigmoid(x);
            }

            return x;
        }
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                // 释放资源
                layers?.Dispose();
                act?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}