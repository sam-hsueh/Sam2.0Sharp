using Sam2Sharp.Modeling.Backbones;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Sam2Sharp.Modeling;

namespace Sam2Sharp.Modeling.Sam2
{
	public class MaskDecoder : Module<(Tensor, Tensor, Tensor, Tensor, bool, bool, List<Tensor>), (Tensor, Tensor, Tensor, Tensor)>
    {
		private readonly Module<Tensor, Tensor, Tensor, (Tensor, Tensor)> transformer;
		private readonly int num_mask_tokens;
		private readonly ModuleList<MLP> output_hypernetworks_mlps;
		private readonly MLP iou_prediction_head;
		private readonly int transformer_dim;
        private readonly int num_multimask_outputs;
        private readonly Embedding iou_token;
        private readonly Embedding mask_tokens;
        private readonly bool pred_obj_scores;
        private readonly Embedding obj_score_token;
        private readonly bool use_multimask_token_for_obj_ptr;
        private readonly Sequential output_upscaling;
        private readonly bool use_high_res_features;
        internal readonly Conv2d conv_s0;
        internal readonly Conv2d conv_s1;
        private readonly Module<Tensor,Tensor> pred_obj_score_head;
        private readonly bool dynamic_multimask_via_stability;
        private readonly float dynamic_multimask_stability_delta;
        private readonly float dynamic_multimask_stability_thresh;
        private readonly Module<Tensor, Tensor> act1 = null;
		private readonly Module<Tensor, Tensor> act2 = null;

        public MaskDecoder(
            int transformer_dim,
            Module<Tensor, Tensor, Tensor, (Tensor, Tensor)> transformer,
            int num_multimask_outputs = 3,
            Func<Module<Tensor, Tensor>> activation = null,
            int iou_head_depth = 3,
            int iou_head_hidden_dim = 256,
            bool use_high_res_features = false,
            bool iou_prediction_use_sigmoid = false,
            bool dynamic_multimask_via_stability = false,
            float dynamic_multimask_stability_delta = 0.05f,
            float dynamic_multimask_stability_thresh = 0.98f,
            bool pred_obj_scores = false,
            bool pred_obj_scores_mlp = false,
            bool use_multimask_token_for_obj_ptr = false) : base("MaskDecoder")
        {
            this.transformer_dim = transformer_dim;
            this.transformer = transformer;
            this.iou_token = Embedding(1, transformer_dim);
            this.num_mask_tokens = num_multimask_outputs + 1;
            this.mask_tokens = Embedding(this.num_mask_tokens, transformer_dim);
            this.num_multimask_outputs = num_multimask_outputs;
            this.pred_obj_scores = pred_obj_scores;
            
            if (this.pred_obj_scores)
                this.obj_score_token = Embedding(1, transformer_dim);
                
            this.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr;

            activation = activation ?? GELU;
            
            output_upscaling = Sequential(
                ConvTranspose2d(transformer_dim, transformer_dim / 4, kernel_size: 2, stride: 2),
				new LayerNorm2d(transformer_dim / 4),
                activation(),
                ConvTranspose2d(transformer_dim / 4, transformer_dim / 8, kernel_size: 2, stride: 2),
                activation()
            );

            this.use_high_res_features = use_high_res_features;
            if (use_high_res_features)
            {
                conv_s0 = Conv2d(transformer_dim, transformer_dim / 8, kernel_size: 1, stride: 1);
                conv_s1 = Conv2d(transformer_dim, transformer_dim / 4, kernel_size: 1, stride: 1);
            }

            this.output_hypernetworks_mlps = new ModuleList<MLP>();
            for (int i = 0; i < this.num_mask_tokens; i++)
            {
                output_hypernetworks_mlps.Add(new MLP(transformer_dim, transformer_dim, transformer_dim / 8, 3));
            }
           // output_hypernetworks_mlps = new ModuleList<MLP>();

			iou_prediction_head = new MLP(
                transformer_dim,
                iou_head_hidden_dim,
                this.num_mask_tokens,
                iou_head_depth,
                sigmoid_output: iou_prediction_use_sigmoid,
                activation:nn.ReLU()
            );

            if (this.pred_obj_scores)
            {
                if (pred_obj_scores_mlp)
                    pred_obj_score_head = new MLP(transformer_dim, transformer_dim, 1, 3);
                else
                    pred_obj_score_head = Linear(transformer_dim, 1);
            }

            this.dynamic_multimask_via_stability = dynamic_multimask_via_stability;
            this.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta;
            this.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh;


			RegisterComponents();
		}

		public override (Tensor, Tensor, Tensor, Tensor) forward((Tensor, Tensor, Tensor, Tensor, bool, bool, List<Tensor>?) input)
        {
            Tensor image_embeddings = input.Item1;
            Tensor image_pe = input.Item2;
            Tensor sparse_prompt_embeddings = input.Item3;
            Tensor dense_prompt_embeddings = input.Item4;
            bool multimask_output = input.Item5;
            bool repeat_image = input.Item6;
            List <Tensor>? high_res_features = input.Item7;

			var (masks, iou_pred, mask_tokens_out, object_score_logits) = PredictMasks(
                image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, repeat_image, high_res_features);

			// Select the correct mask or masks for output
			TensorIndex mask_slice = multimask_output ? TensorIndex.Slice(1, null) : TensorIndex.Slice(0, 1);
			masks = masks[.., mask_slice, .., ..];
			iou_pred = iou_pred[.., mask_slice];

			if (multimask_output)
            {
				masks = masks[.., mask_slice, .., ..];
				iou_pred = iou_pred[.., mask_slice];
			}
			else if (dynamic_multimask_via_stability && !training)
            {
                (masks, iou_pred) = _dynamic_multimask_via_stability(masks, iou_pred);
            }
            else
            {
				masks = masks[.., mask_slice, .., ..];
				iou_pred = iou_pred[.., mask_slice];
			}

			Tensor sam_tokens_out;
            if (multimask_output && use_multimask_token_for_obj_ptr)
            {
                sam_tokens_out = mask_tokens_out[.., TensorIndex.Slice(1, null)]; // [b, 3, c] shape
            }
            else
            {
                sam_tokens_out = mask_tokens_out[.., TensorIndex.Slice(0, 1)]; // [b, 1, c] shape
            }

            return (masks, iou_pred, sam_tokens_out, object_score_logits);
        }

            public (Tensor, Tensor, Tensor, Tensor) PredictMasks(
            Tensor image_embeddings,
            Tensor image_pe,
            Tensor sparse_prompt_embeddings,
            Tensor dense_prompt_embeddings,
            bool repeat_image,
            List<Tensor> high_res_features = null)
        {
            Tensor output_tokens;
            int s = 0;

            if (pred_obj_scores)
            {
                output_tokens = torch.cat(new[] {
                    obj_score_token.weight!,
                    iou_token.weight!,
                    mask_tokens.weight!
                }, dim: 0);
                s = 1;
            }
            else
            {
                output_tokens = torch.cat(new[] {
                    iou_token.weight!,
                    mask_tokens.weight!
                }, dim: 0);
            }

            output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1);
            var tokens = torch.cat(new[] { output_tokens, sparse_prompt_embeddings }, dim: 1);

            Tensor src;
            if (repeat_image)
                src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim: 0);
            else
            {
                if (image_embeddings.shape[0] != tokens.shape[0])
                    throw new ArgumentException("image_embeddings and tokens batch size mismatch");
                src = image_embeddings;
            }

            src = src + dense_prompt_embeddings;
            
            if (image_pe.size(0) != 1)
                throw new ArgumentException("image_pe should have size 1 in batch dim (from `get_dense_pe()`)");
                
            var pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim: 0);
            var (b, c, h, w) = (src.shape[0], src.shape[1], src.shape[2], src.shape[3]);

            // Run the transformer
            var (hs, srcTransformed) = ((Tensor, Tensor))transformer.forward(src, pos_src, tokens);
            var iou_token_out = hs[.., s, ..];
            var mask_tokens_out = hs[.., TensorIndex.Slice(s + 1, s + 1 + num_mask_tokens), ..];

            // Upscale mask embeddings and predict masks using the mask tokens
            srcTransformed = srcTransformed.transpose(1, 2).view(b, c, h, w);
            
            Tensor upscaled_embedding;
            if (!use_high_res_features)
            {
                upscaled_embedding = output_upscaling.forward(srcTransformed);
            }
            else
            {
                if (high_res_features == null || high_res_features.Count < 2)
                    throw new ArgumentException("high_res_features must contain at least two elements");
                    
                var feat_s0 = high_res_features[0];
                var feat_s1 = high_res_features[1];
                
                var dc1 = (ConvTranspose2d)output_upscaling[0];
                var ln1 = (LayerNorm2d)output_upscaling[1];
                var act1 = (Module<Tensor,Tensor>)output_upscaling[2];
                var dc2 = (ConvTranspose2d)output_upscaling[3];
                var act2 = (Module<Tensor, Tensor>)output_upscaling[4];
                
                var up1 = dc1.forward(srcTransformed) + feat_s1;
                up1 = act1.forward(ln1.forward(up1));
                
                upscaled_embedding = dc2.forward(up1) + feat_s0;
                upscaled_embedding = act2.forward(upscaled_embedding);
            }

            var hyperInList = new List<Tensor>();
            for (int i = 0; i < num_mask_tokens; i++)
            {
                hyperInList.Add(output_hypernetworks_mlps[i].forward(mask_tokens_out[.., i, ..]));
            }
            
            var hyperIn = torch.stack(hyperInList, dim: 1);
            var (b2, c2, h2, w2) = (upscaled_embedding.shape[0], upscaled_embedding.shape[1], 
                                    upscaled_embedding.shape[2], upscaled_embedding.shape[3]);
                                    
            var masks = hyperIn.matmul(upscaled_embedding.view(b2, c2, h2 * w2)).view(b2, -1, h2, w2);

            // Generate mask quality predictions
            var iou_pred = iou_prediction_head.forward(iou_token_out);
            Tensor object_score_logits;
            
            if (pred_obj_scores)
            {
                object_score_logits = pred_obj_score_head.forward(hs[.., 0, ..]);
            }
            else
            {
                // Default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
                object_score_logits = 10.0f * torch.ones(iou_pred.shape[0], 1, device: iou_pred.device);
            }

            return (masks, iou_pred, mask_tokens_out, object_score_logits);
        }

        private Tensor _get_stability_scores(Tensor mask_logits)
        {
            mask_logits = mask_logits.flatten(-2);
            var stability_delta = dynamic_multimask_stability_delta;
            var area_i = mask_logits.gt(stability_delta).sum(-1).to(torch.float32);
            var area_u = mask_logits.gt(-stability_delta).sum(-1).to(torch.float32);
            return torch.where(area_u.gt(0), area_i / area_u, torch.tensor(1.0f, device: mask_logits.device));
        }

        private (Tensor, Tensor) _dynamic_multimask_via_stability(Tensor all_mask_logits, Tensor all_iou_scores)
        {
            // The best mask from multimask output tokens (1~3)
            var multimask_logits = all_mask_logits[.., TensorIndex.Slice(1), .., ..];
            var multimask_iou_scores = all_iou_scores[.., TensorIndex.Slice(1)];
            var best_scores_inds = torch.argmax(multimask_iou_scores, dim: -1);
            var batch_inds = torch.arange(multimask_iou_scores.size(0), device: all_iou_scores.device);
            
            var best_multimask_logits = multimask_logits[batch_inds, best_scores_inds].unsqueeze(1);
            var best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds].unsqueeze(1);

            // The mask from singlemask output token 0 and its stability score
            var singlemask_logits = all_mask_logits[.., TensorIndex.Slice(0, 1), .., ..];
            var singlemask_iou_scores = all_iou_scores[.., TensorIndex.Slice(0, 1)];
            var stability_scores = _get_stability_scores(singlemask_logits);
            var is_stable = stability_scores.ge(dynamic_multimask_stability_thresh);

            // Dynamically fall back to best multimask output upon low stability scores
            var mask_logits_out = torch.where(
                is_stable.unsqueeze(-1).unsqueeze(-1).expand_as(singlemask_logits),
                singlemask_logits,
                best_multimask_logits
            );
            
            var iou_scores_out = torch.where(
                is_stable.unsqueeze(-1).expand_as(singlemask_iou_scores),
                singlemask_iou_scores,
                best_multimask_iou_scores
            );

            return (mask_logits_out, iou_scores_out);
        }
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                iou_token?.Dispose();
                mask_tokens?.Dispose();
                obj_score_token?.Dispose();
                output_upscaling?.Dispose();
                conv_s0?.Dispose();
                conv_s1?.Dispose();
                pred_obj_score_head?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}