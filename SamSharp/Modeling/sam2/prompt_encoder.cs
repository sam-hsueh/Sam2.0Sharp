using OpenCvSharp;
using Sam2Sharp.Modeling.PositionEncoding;
using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Sam2Sharp.Modeling.Sam2
{
    public class PromptEncoder : Module<(Tensor, Tensor)?,Tensor?, Tensor, (Tensor, Tensor)>
    {
		private readonly int embed_dim;
		private readonly (int, int) image_embedding_size;
		private readonly (int, int) input_image_size;
		private readonly int mask_in_chans;
		private readonly PositionEmbeddingRandom pe_layer;
		private readonly int num_point_embeddings;
		private readonly ModuleList<Embedding> point_embeddings;
		private readonly Embedding not_a_point_embed;
		public (int, int) mask_input_size;
		private readonly Sequential mask_downscaling;
		private readonly Embedding no_mask_embed;

		//private readonly int embed_dim;
  //      private readonly (int, int) input_image_size;
  //      private readonly (int, int) image_embedding_size;
  //      private readonly PositionEmbeddingRandom pe_layer;
  //      private readonly ModuleList<Embedding> point_embeddings;
  //      private readonly Embedding not_a_point_embed;
  //      private readonly Sequential mask_downscaling;
  //      private readonly Embedding no_mask_embed;
  //      private readonly int num_point_embeddings = 4;

        public PromptEncoder(int embed_dim, (int, int) image_embedding_size, (int, int) input_image_size, int mask_in_chans, Func<Module<Tensor, Tensor>> activation = null) : base("PromptEncoder")
        {
			this.embed_dim = embed_dim;
			this.image_embedding_size = image_embedding_size;
			this.mask_input_size = (4 * image_embedding_size.Item1, 4 * image_embedding_size.Item2);
			this.input_image_size = input_image_size;
			this.mask_in_chans = mask_in_chans;
			this.pe_layer = new PositionEmbeddingRandom(embed_dim / 2, 0.1f);

			this.num_point_embeddings = 4;
			point_embeddings = new ModuleList<Embedding>();
			for (int i = 0; i < num_point_embeddings; i++)
			{
				point_embeddings.append(Embedding(1, embed_dim));
			}

			this.not_a_point_embed = nn.Embedding(1, embed_dim);

			this.mask_downscaling = nn.Sequential(
				   nn.Conv2d(1, mask_in_chans / 4, kernel_size: 2, stride: 2),
				   new LayerNorm2d(mask_in_chans / 4),
				   GELU(),
				   nn.Conv2d(mask_in_chans / 4, mask_in_chans, kernel_size: 2, stride: 2),
				   new LayerNorm2d(mask_in_chans),
				   GELU(),
				   nn.Conv2d(mask_in_chans, embed_dim, kernel_size: 1));

			this.no_mask_embed = nn.Embedding(1, embed_dim);
			RegisterComponents();
        }

        public Tensor GetDensePe()
        {
           // return pe_layer.Call(image_embedding_size).unsqueeze(0);
			return this.pe_layer.forward(this.image_embedding_size).unsqueeze(0);
		}

        /// <summary>
        /// Returns the positional encoding used to encode point prompts,
        /// applied to a dense set of points the shape of the image encoding.
        /// </summary>
        /// <returns>Positional encoding with shape 1x(embed_dim)X(embedding_h)X(embedding_w)</returns>
        public Tensor get_dense_pe()
        {
            return this.pe_layer.forward(this.image_embedding_size).unsqueeze(0);
        }


        private Tensor embed_points(Tensor points, Tensor labels, bool pad)
        {
			using var _ = NewDisposeScope();

            points = points + 0.5f;  // Shift to center of pixel
			if (pad)
			{
				Tensor padding_point = torch.zeros(new long[] { points.shape[0], 1, 2 }, device: points.device, dtype: points.dtype);
				Tensor padding_label = -torch.ones(new long[] { labels.shape[0], 1 }, device: labels.device, dtype: labels.dtype);
				points = torch.cat(new Tensor[] { points, padding_point }, dim: 1);
				labels = torch.cat(new Tensor[] { labels, padding_label }, dim: 1);
			}
			Tensor point_embedding = this.pe_layer.forward_with_coords(points, this.input_image_size);
            //long start = DateTime.Now.Ticks;
			//point_embedding[labels == -1] = 0.0f;
			//point_embedding[labels == -1] += this.not_a_point_embed.weight!;
			//point_embedding[labels == 0] += this.point_embeddings[0].weight!;
			//point_embedding[labels == 1] += this.point_embeddings[1].weight!;
            point_embedding = torch.where((labels == -1).unsqueeze(-1),
            torch.zeros_like(point_embedding) + this.not_a_point_embed.weight, point_embedding);
			point_embedding = torch.where((labels == 0).unsqueeze(-1), point_embedding + this.point_embeddings[0].weight, point_embedding);
			point_embedding = torch.where((labels == 1).unsqueeze(-1), point_embedding + this.point_embeddings[1].weight, point_embedding);
			point_embedding = torch.where((labels == 2).unsqueeze(-1), point_embedding + this.point_embeddings[2].weight, point_embedding);
			point_embedding = torch.where((labels == 3).unsqueeze(-1), point_embedding + this.point_embeddings[3].weight, point_embedding);
            //long end = DateTime.Now.Ticks;
            //long GIelapsedMs = (end - start) / TimeSpan.TicksPerMillisecond;
            return point_embedding.MoveToOuterDisposeScope();
        }

        private Tensor EmbedBoxes(Tensor boxes)
        {
            //boxes = boxes + 0.5f; // Shift to center of pixel
            //var coords = boxes.reshape(-1, 2, 2);
            //var cornerEmbedding = pe_layer.forward_with_coords(coords, input_image_size);
            //cornerEmbedding[.., 0, ..] += point_embeddings[2].weight;
            //cornerEmbedding[.., 1, ..] += point_embeddings[3].weight;
            //return cornerEmbedding;

			using var _ = NewDisposeScope();
			boxes = boxes + 0.5f;  // Shift to center of pixel
			Tensor coords = boxes.reshape(-1, 2, 2);
			Tensor corner_embedding = this.pe_layer.forward_with_coords(coords, this.input_image_size);
			corner_embedding[.., 0, ..] += this.point_embeddings[2].weight!;
			corner_embedding[.., 1, ..] += this.point_embeddings[3].weight!;
			return corner_embedding.MoveToOuterDisposeScope();

		}

		private Tensor EmbedMasks(Tensor masks)
        {
            return mask_downscaling.forward(masks);
        }

        private long GetBatchSize((Tensor, Tensor)? points, Tensor boxes, Tensor masks)
        {
            if (points.Value.Item1 is not null)
                return points.Value.Item1.shape[0];
            if (boxes is not null)
                return boxes.shape[0];
            if (masks is not null)
                return masks.shape[0];
            return 1;
        }

		private torch.Device GetDevice()
		{
			return point_embeddings[0].weight.device;
		}
		public override (Tensor, Tensor) forward((Tensor, Tensor)? points, Tensor? boxes, Tensor? masks)
		{
            //long start = DateTime.Now.Ticks;
			using var _ = NewDisposeScope();
            (Device device, ScalarType dtype) = Common.GetDeviceAndScaleType(this);
			//Debug.WriteLine($"Mask Decoder Time: {GIelapsedMs} ms");

			long bs = this.GetBatchSize(points, boxes, masks);

			Tensor sparse_embeddings = torch.empty(new long[] { bs, 0, this.embed_dim }, device: device, dtype: dtype);

			if (points.Value.Item1 is not null)
			{
				Tensor coords = points.Value.Item1/*.to(dtype, device)*/;
				Tensor labels = points.Value.Item2/*.to(dtype, device)*/;
                Tensor point_embeddings = this.embed_points(coords, labels, pad: (boxes is null));
				sparse_embeddings = torch.cat(new Tensor[] { sparse_embeddings, point_embeddings }, dim: 1);
			}

			if (boxes is not null)
			{
				Tensor box_embeddings = this.EmbedBoxes(boxes);
				if (sparse_embeddings.shape[0] == 1)
				{
					sparse_embeddings = sparse_embeddings.repeat(box_embeddings.shape[0], 1, 1);
				}
				sparse_embeddings = torch.cat(new Tensor[] { sparse_embeddings, box_embeddings }, dim: 1);
			}
			Tensor dense_embeddings = (masks is not null) ? this.EmbedMasks(masks) : this.no_mask_embed.weight!.reshape(1, -1, 1, 1).expand(bs, -1, this.image_embedding_size.Item1, this.image_embedding_size.Item2);
			//long end = DateTime.Now.Ticks;
			//long GIelapsedMs = (end - start) / TimeSpan.TicksPerMillisecond;
			return (sparse_embeddings.MoveToOuterDisposeScope(), dense_embeddings.MoveToOuterDisposeScope());
		}
		public void Dispose(bool disposing)
		{
			if (disposing)
			{
				pe_layer?.Dispose();
				foreach (var embedding in point_embeddings)
				{
					embedding?.Dispose();
				}
				not_a_point_embed?.Dispose();
				mask_downscaling?.Dispose();
				no_mask_embed?.Dispose();
			}
			base.Dispose(disposing);
        }
        //public override (Tensor, Tensor) forward((Tensor pointsCoords, Tensor pointsLabels), points, (Tensor boxes, Tensor masks))
        //      {
        //          var points = pointsCoords is not null && pointsLabels is not null ? (pointsCoords, pointsLabels) : ((Tensor, Tensor)?)null;
        //          var bs = GetBatchSize(points, boxes, masks);

        //          var sparseEmbeddings = torch.empty(bs, 0, embed_dim, device: GetDevice());

        //          if (points.HasValue)
        //          {
        //              var (coords, labels) = points.Value;
        //              var pointEmbeddings = embed_points(coords, labels, pad: boxes is null);
        //              sparseEmbeddings = torch.cat(new[] { sparseEmbeddings, pointEmbeddings }, dim: 1);
        //          }

        //          if (boxes is not null)
        //          {
        //              var boxEmbeddings = EmbedBoxes(boxes);
        //              sparseEmbeddings = torch.cat(new[] { sparseEmbeddings, boxEmbeddings }, dim: 1);
        //          }

        //          Tensor denseEmbeddings;
        //          if (masks is not null)
        //          {
        //              denseEmbeddings = EmbedMasks(masks);
        //          }
        //          else
        //          {
        //              denseEmbeddings = no_mask_embed.weight.reshape(1, -1, 1, 1)
        //                  .expand(bs, -1, image_embedding_size.Item1, image_embedding_size.Item2);
        //          }

        //          return (sparseEmbeddings, denseEmbeddings);
        //      }
    }
}