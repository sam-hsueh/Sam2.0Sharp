using TorchSharp;
using TorchSharp.Modules;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Sam2Sharp.Modeling.PositionEncoding
{
    public class PositionEmbeddingSine : Module<Tensor, Tensor>
    {
        private int num_pos_feats;
        private int temperature;
        private bool normalize;
        private float scale;
        private readonly Dictionary<(int, int), Tensor> cache = new Dictionary<(int, int), Tensor>();

        public PositionEmbeddingSine(int num_pos_feats, int temperature = 10000, bool normalize = true, float? scale = null,
                                     bool warmup_cache = true, int image_size = 1024, int[] strides = null) : base("PositionEmbeddingSine")
        {
            if (num_pos_feats % 2 != 0)
                throw new ArgumentException("Expecting even model width", nameof(num_pos_feats));
            this.num_pos_feats = num_pos_feats / 2;
            this.temperature = temperature;
            this.normalize = normalize;

            if (scale.HasValue && !normalize)
                throw new ArgumentException("Normalize should be True if scale is passed");

            this.scale = scale ?? (2 * (float)Math.PI);

            if (warmup_cache && torch.cuda.is_available())
            {
                var device = torch.CUDA;
                int[] defaultStrides = [ 4, 8, 16, 32 ];
                var useStrides = strides ?? defaultStrides;

                foreach (var stride in useStrides)
                {
                    var cacheKey = (image_size / stride, image_size / stride);
                    _pe(1, device, cacheKey.Item1, cacheKey.Item2);
                }
            }

            RegisterComponents();
        }

        private (Tensor, Tensor) _encode_xy(Tensor x, Tensor y)
        {
            if (x.shape[0] != y.shape[0] || x.Dimensions != 1 || y.Dimensions != 1)
                throw new ArgumentException("Invalid input dimensions for x and y");

            var x_embed = x * scale;
            var y_embed = y * scale;

            var dim_t = arange(num_pos_feats, dtype: ScalarType.Float32, device: x.device);
            dim_t = pow(torch.tensor(temperature, dtype: dim_t.dtype, device: dim_t.device),
                       2 * (dim_t / 2) / num_pos_feats);

            var pos_x = x_embed.unsqueeze(1) / dim_t;
            var pos_y = y_embed.unsqueeze(1) / dim_t;

            pos_x = stack(new[] { pos_x.narrow(1, 0, pos_x.size(1)).sin(), pos_x.narrow(1, 1, pos_x.size(1) - 1).cos() }, 2).flatten(1);
            pos_y = stack(new[] { pos_y.narrow(1, 0, pos_y.size(1)).sin(), pos_y.narrow(1, 1, pos_y.size(1) - 1).cos() }, 2).flatten(1);

            return (pos_x, pos_y);
        }

        public Tensor encode_boxes(Tensor x, Tensor y, Tensor w, Tensor h)
        {
            using var _ = no_grad();
            var (pos_x, pos_y) = _encode_xy(x, y);
            return cat(new[] { pos_y, pos_x, h.unsqueeze(1), w.unsqueeze(1) }, 1);
        }

        public Tensor encode(Tensor x, Tensor y, Tensor w, Tensor h) => encode_boxes(x, y, w, h);

        public Tensor encode_points(Tensor x, Tensor y, Tensor labels)
        {
            using var _ = no_grad();
            var (bx, nx) = (x.shape[0], x.shape[1]);
            var (by, ny) = (y.shape[0], y.shape[1]);
            var (bl, nl) = (labels.shape[0], labels.shape[1]);

            if (bx != by || nx != ny || bx != bl || nx != nl)
                throw new ArgumentException("Input dimensions do not match");

            var (pos_x, pos_y) = _encode_xy(x.flatten(), y.flatten());
            pos_x = pos_x.reshape(bx, nx, -1);
            pos_y = pos_y.reshape(by, ny, -1);

            return cat(new[] { pos_y, pos_x, labels.unsqueeze(2) }, 2);
            //// 1. 解包形状
            //long bx = x.shape[0], nx = x.shape[1];
            //long by = y.shape[0], ny = y.shape[1];
            //long bl = labels.shape[0], nl = labels.shape[1];

            //// 2. 维度断言
            //if (!(bx == by && nx == ny && bx == bl && nx == nl))
            //{
            //    throw new ArgumentException($"维度不匹配：x=({bx},{nx}), y=({by},{ny}), labels=({bl},{nl})");
            //}

            //// 3. 展平+编码+恢复维度
            //var x_flat = x.flatten();
            //var y_flat = y.flatten();
            //(var pos_x_flat, var pos_y_flat) = _encode_xy(x_flat, y_flat);
            //var pos_x = pos_x_flat.reshape(bx, nx, -1);
            //var pos_y = pos_y_flat.reshape(by, ny, -1);

            //// 4. 扩展labels并拼接
            //var labels_expanded = labels.unsqueeze(2);
            //return torch.cat(new[] { pos_y, pos_x, labels_expanded }, dim: 2);             
        }

        private Tensor _pe(int B, Device device, int H, int W)
        {
            using var _ = no_grad();
            var cacheKey = (H, W);
            if (cache.TryGetValue(cacheKey, out var cached))
                return cached.to(device).unsqueeze(0).repeat(B, 1, 1, 1);

            var y_embed = arange(1, H + 1, dtype: ScalarType.Float32, device: device)
                .view(1, -1, 1)
                .repeat(B, 1, W);

            var x_embed = arange(1, W + 1, dtype: ScalarType.Float32, device: device)
                .view(1, 1, -1)
                .repeat(B, H, 1);

            if (normalize)
            {
                const float eps = 1e-6f;
                y_embed = y_embed / (y_embed.narrow(1, H - 1, 1) + eps) * scale;
                x_embed = x_embed / (x_embed.narrow(2, W - 1, 1) + eps) * scale;
            }

            var dim_t = arange(num_pos_feats, dtype: ScalarType.Float32, device: device);
            dim_t = pow(torch.tensor(temperature, dtype: dim_t.dtype, device: dim_t.device),
                       2 * (dim_t / 2) / num_pos_feats);

            //var pos_x = x_embed.unsqueeze(3) / dim_t;
            //var pos_y = y_embed.unsqueeze(3) / dim_t;

            var pos_x = x_embed[.., .., .., TensorIndex.None] / dim_t;
            var pos_y = y_embed[.., .., .., TensorIndex.None] / dim_t;


            pos_x = stack(new[] { pos_x[.., .., .., 0..2].sin(), pos_x[.., .., .., 0..2].cos() }, 4).flatten(3);
            pos_y = stack(new[] { pos_y[.., .., .., 0..2].sin(), pos_y[.., .., .., 0..2].cos() }, 4).flatten(3);

            var pos = cat(new[] { pos_y, pos_x }, 3).permute(0, 3, 1, 2);
            cache[cacheKey] = pos[0];

            return pos;            
        }


        public override Tensor forward(Tensor x)
        {
            using var _ = no_grad();
            int B = (int)x.shape[0];
            int H = (int)x.shape[x.shape.Length - 2];
            int W = (int)x.shape[x.shape.Length - 1];
            return _pe(B, x.device, H, W);
        }
    }


    public class PositionEmbeddingRandom : Module<(int, int), Tensor>
    {
        private readonly Tensor positional_encoding_gaussian_matrix;
        public PositionEmbeddingRandom(int num_pos_feats = 64, float scale = 0) : base(nameof(PositionEmbeddingRandom))
        {
            scale = scale <= 0 ? 1.0f : scale;
            positional_encoding_gaussian_matrix = scale * torch.randn(new long[] { 2, num_pos_feats });
            RegisterComponents();
        }

        //Positionally encode points that are normalized to [0,1].
        private Tensor _pe_encoding(Tensor coords)
        {
            // assuming coords are in [0, 1]^2 square and have d_1 X ... X d_n X 2 shape
            coords = 2 * coords - 1;
            coords = coords.matmul(this.positional_encoding_gaussian_matrix);
            coords = 2 * Math.PI * coords;
            // outputs d_1 X ... X d_n X C shape
            return torch.cat(new Tensor[] { torch.sin(coords), torch.cos(coords) }, dim: -1);
        }

        // Generate positional encoding for a grid of the specified size.
        public override Tensor forward((int, int) size)
        {
            using var _ = NewDisposeScope();
            (int h, int w) = size;
            Device device = this.positional_encoding_gaussian_matrix.device;
            ScalarType dtype = this.positional_encoding_gaussian_matrix.dtype;
            Tensor grid = torch.ones(new long[] { h, w }, device: device, dtype: dtype);
            Tensor y_embed = grid.cumsum(dim: 0) - 0.5;
            Tensor x_embed = grid.cumsum(dim: 1) - 0.5;
            y_embed = y_embed / h;
            x_embed = x_embed / w;

            Tensor pe = this._pe_encoding(torch.stack(new Tensor[] { x_embed, y_embed }, dim: -1).to(dtype, device));

            return pe.permute(2, 0, 1).MoveToOuterDisposeScope();  // C X H X W
        }

        // Positionally encode points that are not normalized to [0,1].
        public Tensor forward_with_coords(Tensor coords_input, (int, int) image_size)
        {
            Tensor coords = coords_input.clone();
            coords[.., .., 0] = coords[.., .., 0] / image_size.Item2;
            coords[.., .., 1] = coords[.., .., 1] / image_size.Item1;
            return this._pe_encoding(coords); // B X N X C
        }
    }

    public static class RotaryPositionalEncoding
    {
        public static (Tensor, Tensor) init_t_xy(int end_x, int end_y)
        {
            var t = torch.arange(end_x * end_y, dtype: ScalarType.Float32);
            var t_x = (t % end_x).to(ScalarType.Float32);
            var t_y = t.div(end_x, RoundingMode.floor).to(ScalarType.Float32);
            return (t_x, t_y);
        }
    //    def init_t_xy(end_x: int, end_y: int):
    //t = torch.arange(end_x* end_y, dtype=torch.float32)
    //t_x = (t % end_x).float ()
    //t_y = torch.div(t, end_x, rounding_mode = "floor").float ()
    //return t_x, t_y

        public static Tensor compute_axial_cis(int dim, int end_x, int end_y, float theta = 10000.0f)
        {
            var a = arange(0, dim, 4).narrow(0, 0, dim / 4).to(ScalarType.Float32) / dim;
            //var dd = a.to(ScalarType.Float32);
            //var array = dd.data<float>().ToArray();
            //a = torch.arange(0, dim, 4)[..(dim / 4)].to(ScalarType.Float32) / dim;
            //dd = a.to(ScalarType.Float32);
            //array = dd.data<float>().ToArray();

            var freqs_x = 1.0f / pow(theta, a);
            //dd = freqs_x.to(ScalarType.Float32);
            //array = dd.data<float>().ToArray();

            var freqs_y = 1.0f / pow(theta, arange(0, dim, 4).narrow(0, 0, dim / 4).to(ScalarType.Float32) / dim);

            var (t_x, t_y) = init_t_xy(end_x, end_y);
            var freqs_x_outer = t_x.unsqueeze(1).matmul(freqs_x.unsqueeze(0));
            var freqs_y_outer = t_y.unsqueeze(1).matmul(freqs_y.unsqueeze(0));

            var freqs_cis_x = torch.polar(ones_like(freqs_x_outer), freqs_x_outer);
            var freqs_cis_y = torch.polar(ones_like(freqs_y_outer), freqs_y_outer);

            return cat(new[] { freqs_cis_x, freqs_cis_y }, -1);
    //        var a = torch.arange(0, dim, 4)[..(dim / 4)].to(ScalarType.Float32) / dim;
    //var freqs_x = 1.0 / (theta**(a));
    //freqs_y = 1.0 / (theta**(torch.arange(0, dim, 4)[..(dim / 4)].to(ScalarType.Float32) / dim));

    //t_x, t_y = init_t_xy(end_x, end_y)
    //freqs_x = torch.outer(t_x, freqs_x)
    //freqs_y = torch.outer(t_y, freqs_y)
    //freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    //freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    //return torch.cat([freqs_cis_x, freqs_cis_y], dim = -1)
       }
    //    def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0) :
    //freqs_x = 1.0 / (theta**(torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    //freqs_y = 1.0 / (theta**(torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    //t_x, t_y = init_t_xy(end_x, end_y)
    //freqs_x = torch.outer(t_x, freqs_x)
    //freqs_y = torch.outer(t_y, freqs_y)
    //freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    //freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    //return torch.cat([freqs_cis_x, freqs_cis_y], dim = -1)

        public static Tensor reshape_for_broadcast(Tensor freqs_cis, Tensor x)
        {
            var ndim = x.Dimensions;
            if (ndim < 2)
                throw new ArgumentException("x must have at least 2 dimensions");

            if (freqs_cis.shape[0] != x.shape[x.Dimensions - 2] || freqs_cis.shape[1] != x.shape[x.Dimensions - 1])
                throw new ArgumentException("freqs_cis shape does not match x shape");

            var shape = new long[ndim];
            for (int i = 0; i < ndim; i++)
                shape[i] = (i >= ndim - 2) ? x.shape[i] : 1;

            return freqs_cis.view(shape);
        }
        //    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        //ndim = x.ndim
        //assert 0 <= 1 < ndim
        //assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
        //shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
        //return freqs_cis.view(* shape)


        public static (Tensor, Tensor) apply_rotary_enc(Tensor xq, Tensor xk, Tensor freqs_cis, bool repeat_freqs_k = false)
        {
            var xqShape = xq.shape.ToArray();
            var newXqShape = xqShape.Take(xqShape.Length - 1).Concat(new[] { xqShape.Last() / 2, 2 }).ToArray();
            var xqComplex = view_as_complex(xq.to(ScalarType.Float32).reshape(newXqShape));

            Tensor xkComplex = null;
            if (xk.shape[xk.Dimensions - 2] != 0)
            {
                var xkShape = xk.shape.ToArray();
                var newXkShape = xkShape.Take(xkShape.Length - 1).Concat(new[] { xkShape.Last() / 2, 2 }).ToArray();
                xkComplex = view_as_complex(xk.to(ScalarType.Float32).reshape(newXkShape));
            }

            var freqsCisBroadcast = reshape_for_broadcast(freqs_cis, xqComplex);
            var xqOut = view_as_real(xqComplex * freqsCisBroadcast).flatten(3);
            xqOut = xqOut.to(xq.dtype).to(xq.device);

            if (xkComplex is null)
                return (xqOut, xk);

            if (repeat_freqs_k)
            {
                var r = xkComplex.shape[xkComplex.Dimensions - 2] / xqComplex.shape[xqComplex.Dimensions - 2];
                if (freqsCisBroadcast.device.type == DeviceType.CUDA)
                {
                    var repeatDims = Enumerable.Repeat(1L, (int)(freqsCisBroadcast.Dimensions) - 2).Concat(new[] { r, 1 }).ToArray();
                    freqsCisBroadcast = freqsCisBroadcast.repeat(repeatDims);
                }
                else
                {
                    freqsCisBroadcast = freqsCisBroadcast.unsqueeze(-3).expand(-1, -1, r, -1, -1).flatten(-3, -2);
                }
            }

            var xkOut = view_as_real(xkComplex * freqsCisBroadcast).flatten(3);
            xkOut = xkOut.to(xk.dtype).to(xk.device);

            return (xqOut, xkOut);
        }
//def apply_rotary_enc(
//    xq: torch.Tensor,
//    xk: torch.Tensor,
//    freqs_cis: torch.Tensor,
//    repeat_freqs_k: bool = False,
//) :
//    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
//    xk_ = (
//        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
//        if xk.shape[-2] != 0
//        else None
//    )
//    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
//    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
//    if xk_ is None:
//        # no keys to rotate, due to dropout
//        return xq_out.type_as(xq).to(xq.device), xk
//    # repeat freqs along seq_len dim to match k seq_len
//    if repeat_freqs_k:
//        r = xk_.shape[-2] // xq_.shape[-2]
//        if freqs_cis.is_cuda:
//            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
//        else:
//            # torch.repeat on complex numbers may not be supported on non-CUDA devices
//            # (freqs_cis has 4 dims and we repeat on dim 2) so we use expand + flatten
//            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
//    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
//    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
    }
}




