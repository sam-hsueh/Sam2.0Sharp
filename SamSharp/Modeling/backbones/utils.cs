using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Sam2Sharp.Modeling.Backbones
{
    public static class Utils
    {
        /// <summary>
        /// Partition into non-overlapping windows with padding if needed.
        /// </summary>
        /// <param name="x">input tokens with [B, H, W, C].</param>
        /// <param name="window_size">window size.</param>
        /// <returns>windows:windows after partition with [B * num_windows, window_size, window_size, C].(Hp, Wp): padded height and width before partition</returns>
        //private static (Tensor, (int, int)) window_partition(Tensor x, int window_size)
        //{
        //    using var _ = NewDisposeScope();
        //    int B = (int)x.shape[0];
        //    int H = (int)x.shape[1];
        //    int W = (int)x.shape[2];
        //    int C = (int)x.shape[3];

        //    int pad_h = (window_size - H % window_size) % window_size;
        //    int pad_w = (window_size - W % window_size) % window_size;

        //    if (pad_h > 0 || pad_w > 0)
        //    {
        //        x = functional.pad(x, new long[] { 0, 0, 0, pad_w, 0, pad_h });
        //    }
        //    (int Hp, int Wp) = (H + pad_h, W + pad_w);
        //    x = x.view(B, Hp / window_size, window_size, Wp / window_size, window_size, C);

        //    Tensor windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C);
        //    return (windows.MoveToOuterDisposeScope(), (Hp, Wp));
        //}

        ///// <summary>
        ///// Window unpartition into original sequences and removing padding.
        ///// </summary>
        ///// <param name="windows">input tokens with [B * num_windows, window_size, window_size, C].</param>
        ///// <param name="window_size">window size.</param>
        ///// <param name="pad_hw">padded height and width (Hp, Wp).</param>
        ///// <param name="hw">original height and width (H, W) before padding.</param>
        ///// <returns>unpartitioned sequences with [B, H, W, C].</returns>
        //private static Tensor window_unpartition(Tensor windows, int window_size, (int, int) pad_hw, (int, int) hw)
        //{
        //    using var _ = NewDisposeScope();
        //    (int Hp, int Wp) = pad_hw;
        //    (int H, int W) = hw;

        //    int B = (int)windows.shape[0] / (Hp * Wp / window_size / window_size);
        //    Tensor x = windows.view(B, Hp / window_size, Wp / window_size, window_size, window_size, -1);
        //    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1);
        //    if (Hp > H || Wp > W)
        //    {
        //        x = x[.., ..H, ..W, ..].contiguous();
        //    }
        //    return x.MoveToOuterDisposeScope();

        //}     
	}
    /// <summary>  
    /// Image to Patch Embedding.  
    /// </summary>  
    public class PatchEmbed : Module<Tensor, Tensor>
    {
        int kernel_size = 7;
        int stride = 4;
        int padding = 3;
        int in_chans = 3;
        int embed_dim = 768;

        private readonly Conv2d proj;

        /// <summary>
        /// Patch Embedding
        /// </summary>
        /// <param name="kernel_size">kernel size of the projection layer.</param>
        /// <param name="stride">stride of the projection layer.</param>
        /// <param name="padding">padding size of the projection layer.</param>
        /// <param name="in_chans">Number of input image channels.</param>
        /// <param name="embed_dim">Patch embedding dimension.</param>
        internal PatchEmbed(int embed_dim, int in_chans = 3, int kernel_size = 7, int stride = 4, int padding = 3) : base(nameof(PatchEmbed))
        {
            this.in_chans = in_chans;
            this.embed_dim = embed_dim;
            this.kernel_size = kernel_size;
            this.stride = stride;
            this.padding = padding;

            proj = Conv2d(in_channels: in_chans, out_channels: embed_dim, kernel_size: kernel_size, stride: stride, padding: padding);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            x = proj.forward(x);
            // B C H W -> B H W C  
            x = x.permute(0, 2, 3, 1);
            return x;
        }
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                proj?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}