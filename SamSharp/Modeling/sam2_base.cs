using Sam2Sharp.Modeling.Backbones;
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static Sam2Sharp.Modeling.Sam2.Transformer;
using static Sam2Sharp.Utils.Classes;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Sam2Sharp.Modeling
{
    public class Sam2Base : Module<Tensor, Tensor>
    {
        private const float NO_OBJ_SCORE = -1024;
        public readonly Device device;
        public readonly ScalarType dtype;
        private long[] original_size = null;
        private float scaleFactor = 0.0f;

        // Image encoder
        private readonly ImageEncoder image_encoder;
        private readonly bool use_high_res_features_in_sam;
        private readonly int num_feature_levels;
        private readonly bool use_obj_ptrs_in_encoder;
        private readonly int max_obj_ptrs_in_encoder;
        private readonly Conv2d mask_downsample;
        private readonly bool add_tpos_enc_to_obj_ptrs;
        private readonly bool proj_tpos_enc_in_obj_ptrs;
        private readonly bool use_signed_tpos_enc_to_obj_ptrs;
        private readonly bool only_obj_ptrs_in_the_past_for_eval;
        // Memory attention components
        private readonly MemoryAttention memory_attention;
        private readonly int hidden_dim;

        // Memory encoder components
        private readonly MemoryEncoder memory_encoder;
        private int mem_dim;
        private int num_maskmem;
        private readonly Parameter maskmem_tpos_enc;
        public readonly Parameter no_mem_embed;
        private readonly Parameter no_mem_pos_enc;
        public readonly bool directly_add_no_mem_embed;
        private readonly float sigmoid_scale_for_mem_enc;
        private readonly float sigmoid_bias_for_mem_enc;
        private readonly bool binarize_mask_from_pts_for_mem_enc;
        private readonly bool non_overlap_masks_for_mem_enc;
        private readonly int memory_temporal_stride_for_eval;
        private readonly bool use_mask_input_as_output_without_sam;
        private readonly bool multimask_output_in_sam;
        private readonly int multimask_min_pt_num;
        private readonly int multimask_max_pt_num;
        private readonly bool multimask_output_for_tracking;
        private readonly bool use_multimask_token_for_obj_ptr;
        private readonly bool iou_prediction_use_sigmoid;

        // SAM components
        public readonly int image_size;
        private readonly int backbone_stride;
        private readonly Dictionary<string, Tensor> sam_mask_decoder_extra_args;
        private readonly bool pred_obj_scores;
        private readonly bool pred_obj_scores_mlp;
        private readonly bool fixed_no_obj_ptr;
        private readonly bool soft_no_obj_ptr;
        public readonly Parameter no_obj_ptr;
        private readonly bool use_mlp_for_obj_ptr_proj;
        private readonly Parameter no_obj_embed_spatial;
  //      public readonly bool no_obj_embed_spatial_b;

        // SAM heads
        private readonly int sam_prompt_embed_dim;
        private readonly int sam_image_embedding_size;
        public readonly Sam2.PromptEncoder sam_prompt_encoder;
        public readonly Sam2.MaskDecoder sam_mask_decoder;
        //public Sam2Sharp.Modeling.Sam.PromptEncoder sam2Sharp_prompt_encoder
        //{
        //    set; get;
        //}
        //public Sam2Sharp.Modeling.Sam.MaskDecoder sam2Sharp_mask_decoder
        //{
        //    set; get;
        //}
        private readonly Module<Tensor, Tensor> obj_ptr_proj;
        private readonly Module<Tensor, Tensor> obj_ptr_tpos_proj;
        private readonly int max_cond_frames_in_attn;
        private readonly float[] pixel_mean = [0.485f, 0.456f, 0.406f ];
        private readonly float[] pixel_std = [0.229f, 0.224f, 0.225f];

        public Sam2Base(
            ImageEncoder image_encoder,
            MemoryAttention memory_attention,
            MemoryEncoder memory_encoder,
            int num_maskmem = 7,
            int image_size = 512,
            int backbone_stride = 16,
            float sigmoid_scale_for_mem_enc = 1.0f,
            float sigmoid_bias_for_mem_enc = 0.0f,
            bool binarize_mask_from_pts_for_mem_enc = false,
            bool use_mask_input_as_output_without_sam = false,
            int max_cond_frames_in_attn = -1,
            bool directly_add_no_mem_embed = false,
            bool use_high_res_features_in_sam = false,
            bool multimask_output_in_sam = false,
            int multimask_min_pt_num = 1,
            int multimask_max_pt_num = 1,
            bool multimask_output_for_tracking = false,
            bool use_multimask_token_for_obj_ptr = false,
            bool iou_prediction_use_sigmoid = false,
            int memory_temporal_stride_for_eval = 1,
            bool non_overlap_masks_for_mem_enc = false,
            bool use_obj_ptrs_in_encoder = false,
            int max_obj_ptrs_in_encoder = 16,
            bool add_tpos_enc_to_obj_ptrs = true,
            bool proj_tpos_enc_in_obj_ptrs = false,
            bool use_signed_tpos_enc_to_obj_ptrs = false,
            bool only_obj_ptrs_in_the_past_for_eval = false,
            bool pred_obj_scores = false,
            bool pred_obj_scores_mlp = false,
            bool fixed_no_obj_ptr = false,
            bool soft_no_obj_ptr = false,
            bool use_mlp_for_obj_ptr_proj = false,
            bool no_obj_embed_spatial = false,
            Dictionary<string, Tensor> sam_mask_decoder_extra_args = null,
            bool compile_image_encoder = false) : base("Sam2Base")
        {
            // Image backbone
            this.image_encoder = image_encoder;
            this.use_high_res_features_in_sam = use_high_res_features_in_sam;
            this.num_feature_levels = use_high_res_features_in_sam ? 3 : 1;
            this.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder;
            this.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder;

            if (use_obj_ptrs_in_encoder)
            {
                mask_downsample = Conv2d(1, 1, kernel_size: 4, stride: 4);
            }
            // no-op fallback for RegisterComponent call in original code

            this.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs;
            this.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs;
            this.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs;
            this.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval;

            // Memory attention
            this.memory_attention = memory_attention;
            try
            {
                this.hidden_dim = ((FpnNeck)image_encoder.neck).d_model;
            }
            catch
            {
                this.hidden_dim = 256; // fallback
            }

            // Memory encoder
            this.memory_encoder = memory_encoder;
            this.mem_dim = hidden_dim;

            // Attempt to read out_proj weight via reflection if possible
            try
            {
                mem_dim = memory_encoder.out_dim;
                //var out_projProp = memory_encoder.out_proj;
                //if (out_projProp != null)
                //{
                //    var out_projVal = out_projProp.GetValue(memory_encoder);
                //    var weightProp = out_projVal?.GetType().GetProperty("weight");
                //    var weightVal = weightProp?.GetValue(out_projVal) as Tensor;
                //    if (weightVal is not null)
                //    {
                //        mem_dim = (int)weightVal.shape[0];
                //    }
                //}
            }
            catch { /* ignore reflection failure */ }
            this.num_maskmem = num_maskmem;
            maskmem_tpos_enc = Parameter(torch.zeros(num_maskmem, 1, 1, mem_dim));
            // RegisterParameter may not exist in this TorchSharp version; ignore
            // RegisterParameter("maskmem_tpos_enc", maskmem_tpos_enc);
            nn.init.trunc_normal_(maskmem_tpos_enc, std: 0.02f);

            no_mem_embed = Parameter(torch.zeros(1, 1, hidden_dim));
            no_mem_pos_enc = Parameter(torch.zeros(1, 1, hidden_dim));
            // RegisterParameter("no_mem_embed", no_mem_embed);
            // RegisterParameter("no_mem_pos_enc", no_mem_pos_enc);
            nn.init.trunc_normal_(no_mem_embed, std: 0.02f);
            nn.init.trunc_normal_(no_mem_pos_enc, std: 0.02f);

            this.directly_add_no_mem_embed = directly_add_no_mem_embed;
            this.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc;
            this.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc;
            this.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc;
            this.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc;
            this.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval;
            this.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam;
            this.multimask_output_in_sam = multimask_output_in_sam;
            this.multimask_min_pt_num = multimask_min_pt_num;
            this.multimask_max_pt_num = multimask_max_pt_num;
            this.multimask_output_for_tracking = multimask_output_for_tracking;
            this.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr;
            this.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid;

            // SAM components
            this.image_size = image_size;
            this.backbone_stride = backbone_stride;
            this.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args;
            this.pred_obj_scores = pred_obj_scores;
            this.pred_obj_scores_mlp = pred_obj_scores_mlp;
            this.fixed_no_obj_ptr = fixed_no_obj_ptr;
            this.soft_no_obj_ptr = soft_no_obj_ptr;
            if (fixed_no_obj_ptr)
            {
                if (!pred_obj_scores || !use_obj_ptrs_in_encoder)
                    throw new ArgumentException("fixed_no_obj_ptr requires pred_obj_scores and use_obj_ptrs_in_encoder to be true");

                no_obj_ptr = Parameter(torch.zeros(1, hidden_dim));
                // RegisterParameter("no_obj_ptr", no_obj_ptr);
                nn.init.trunc_normal_(no_obj_ptr, std: 0.02f);
            }
            else
            {
                no_obj_ptr = null;
            }

            this.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj;
            this.no_obj_embed_spatial = null;
            if (no_obj_embed_spatial==true)
            {
               this.no_obj_embed_spatial = Parameter(torch.zeros(1, mem_dim));
                nn.init.trunc_normal_(this.no_obj_embed_spatial, std: 0.02f);
            }

            // Build SAM heads
            (sam_prompt_embed_dim, sam_image_embedding_size, sam_prompt_encoder, sam_mask_decoder, obj_ptr_proj, obj_ptr_tpos_proj) = _build_sam_heads();
            // RegisterComponent(sam_prompt_encoder);
            // RegisterComponent(sam_mask_decoder);
            // RegisterComponent(obj_ptr_proj);
            // RegisterComponent(obj_ptr_tpos_proj);

            this.max_cond_frames_in_attn = max_cond_frames_in_attn;
            RegisterComponents();

            // Note: TorchSharp doesn't support module compilation like PyTorch
        }

        private (int, int, Sam2Sharp.Modeling.Sam2.PromptEncoder, Sam2Sharp.Modeling.Sam2.MaskDecoder, Module<Tensor, Tensor>, Module<Tensor, Tensor>) _build_sam_heads()
        {
            var sam_prompt_embed_dim = hidden_dim;
            var sam_image_embedding_size = image_size / backbone_stride;

            var prompt_encoder = new Sam2Sharp.Modeling.Sam2.PromptEncoder(
                input_image_size: (image_size, image_size),
                embed_dim: sam_prompt_embed_dim,
                image_embedding_size: (sam_image_embedding_size, sam_image_embedding_size),
                mask_in_chans: 16
            );

            var transformer = new Sam2Sharp.Modeling.Sam2.Transformer.TwoWayTransformer(
                depth: 2,
                embedding_dim: sam_prompt_embed_dim,
                mlp_dim: 2048,
                num_heads: 8
            );

            var mask_decoder_args = new Dictionary<string, object>
             {
                 { "num_multimask_outputs", 3 },
                 { "transformer", transformer },
                 { "transformer_dim", sam_prompt_embed_dim },
                 { "iou_head_depth", 3 },
                 { "iou_head_hidden_dim", 256 },
                 { "use_high_res_features", use_high_res_features_in_sam },
                 { "iou_prediction_use_sigmoid", iou_prediction_use_sigmoid },
                 { "pred_obj_scores", pred_obj_scores },
                 { "pred_obj_scores_mlp", pred_obj_scores_mlp },
                 { "use_multimask_token_for_obj_ptr", use_multimask_token_for_obj_ptr }
             };

            if (sam_mask_decoder_extra_args != null)
            {
                foreach (var kvp in sam_mask_decoder_extra_args)
                    mask_decoder_args[kvp.Key] = kvp.Value;
            }

            // Construct MaskDecoder with transformer and transformer_dim
            var mask_decoder = new Sam2Sharp.Modeling.Sam2.MaskDecoder(
                transformer_dim: sam_prompt_embed_dim,
                transformer: transformer,
                num_multimask_outputs: (int)mask_decoder_args["num_multimask_outputs"],
                //activation: GELU,
                iou_head_depth: (int)mask_decoder_args["iou_head_depth"],
                iou_head_hidden_dim: (int)mask_decoder_args["iou_head_hidden_dim"],
                use_high_res_features: (bool)mask_decoder_args["use_high_res_features"],
                iou_prediction_use_sigmoid: (bool)mask_decoder_args["iou_prediction_use_sigmoid"],
                pred_obj_scores: (bool)mask_decoder_args["pred_obj_scores"],
                pred_obj_scores_mlp: (bool)mask_decoder_args["pred_obj_scores_mlp"],
                use_multimask_token_for_obj_ptr: (bool)mask_decoder_args["use_multimask_token_for_obj_ptr"]
            );

            Module<Tensor, Tensor> obj_ptr_proj;
            if (use_obj_ptrs_in_encoder)
            {
                if (use_mlp_for_obj_ptr_proj)
                {
                    obj_ptr_proj = new MLP(sam_prompt_embed_dim, sam_prompt_embed_dim, sam_prompt_embed_dim, 3);
                }
                else
                {
                    obj_ptr_proj = Linear(sam_prompt_embed_dim, sam_prompt_embed_dim);
                }
            }
            else
            {
                obj_ptr_proj = Identity();
            }

            Module<Tensor, Tensor> obj_ptr_tpos_proj;
            if (proj_tpos_enc_in_obj_ptrs)
            {
                obj_ptr_tpos_proj = Linear(sam_prompt_embed_dim, mem_dim);
            }
            else
            {
                obj_ptr_tpos_proj = Identity();
            }

            return (sam_prompt_embed_dim, sam_image_embedding_size, prompt_encoder, mask_decoder, obj_ptr_proj, obj_ptr_tpos_proj);
        }

        public override Tensor forward(Tensor input)
        {
            throw new NotImplementedException("Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning. See notebooks/video_predictor_example.ipynb for an inference example.");
        }

        public (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) _forward_sam_heads(Tensor backbone_features, (Tensor, Tensor) point_inputs, Tensor mask_inputs = null, List<Tensor> high_res_features = null, bool multimask_output = false)
        {
            var B = backbone_features.size(0);
            var device = backbone_features.device;

            // Validate input dimensions
            if (backbone_features.size(1) != sam_prompt_embed_dim)
                throw new ArgumentException($"Backbone features have incorrect channel dimension: {backbone_features.size(1)} vs {sam_prompt_embed_dim}");

            if (backbone_features.size(2) != sam_image_embedding_size || backbone_features.size(3) != sam_image_embedding_size)
                throw new ArgumentException($"Backbone features have incorrect spatial dimensions");

            // Handle point prompts
            Tensor sam_point_coords, sam_point_labels;
            if (point_inputs.Item1 is not null)
            {
                sam_point_coords = (Tensor)point_inputs.Item1;
                sam_point_labels = (Tensor)point_inputs.Item2;

                if (sam_point_coords.size(0) != B || sam_point_labels.size(0) != B)
                    throw new ArgumentException("Point inputs have incorrect batch size");
            }
            else
            {
                sam_point_coords = torch.zeros(B, 1, 2, device: device);
                sam_point_labels = -torch.ones(B, 1, dtype: ScalarType.Int32, device: device);
            }

            // Handle mask prompts
            Tensor sam_mask_prompt = null;
            if (mask_inputs is not null)
            {
                if (mask_inputs.ndim != 4 || mask_inputs.size(0) != B || mask_inputs.size(1) != 1)
                    throw new ArgumentException("Mask inputs have incorrect dimensions");

                var targetSize = sam_prompt_encoder.mask_input_size;
                if (mask_inputs.size(2) != targetSize.Item1 || mask_inputs.size(3) != targetSize.Item2)
                {
                    sam_mask_prompt = torch.nn.functional.interpolate(
                        mask_inputs.to(ScalarType.Float32),
                        size: [targetSize.Item1, targetSize.Item2],
                        mode: InterpolationMode.Bilinear,
                        align_corners: false,
                        antialias: true
                    );
                }
                else
                {
                    sam_mask_prompt = mask_inputs;
                }
            }

            // Encode prompts
            var (sparse_embeddings, dense_embeddings) = sam_prompt_encoder.forward(
                points: (sam_point_coords, sam_point_labels),
                boxes: null,
                masks: sam_mask_prompt
            );

            // Decode masks
            var decoderOutput = sam_mask_decoder.forward(
                (image_embeddings: backbone_features,
                image_pe: sam_prompt_encoder.GetDensePe(),
                sparse_prompt_embeddings: sparse_embeddings,
                dense_prompt_embeddings: dense_embeddings,
                multimask_output: multimask_output,
                repeat_image: false,
                high_res_features: high_res_features)
            );

            var (low_res_multimasks, ious, sam_output_tokens, object_score_logits) = decoderOutput;

            // Handle object scores
            if (pred_obj_scores)
            {
                var is_obj_appearing = object_score_logits > 0;
                low_res_multimasks = torch.where(
                    is_obj_appearing.unsqueeze(1).unsqueeze(1),
                    low_res_multimasks,
                    torch.tensor(NO_OBJ_SCORE, device: device)
                );
            }

            // Convert to float32 and upsample
            low_res_multimasks = low_res_multimasks.to(ScalarType.Float32);
            var high_res_multimasks = torch.nn.functional.interpolate(
                low_res_multimasks,
                size: [image_size, image_size],
                mode: InterpolationMode.Bilinear,
                align_corners: false
            );

            // Select best mask
            Tensor low_res_masks, high_res_masks;
            Tensor sam_output_token = sam_output_tokens.index_select(1, torch.tensor(0, device: device));

            if (multimask_output)
            {
                var best_iou_inds = torch.argmax(ious, dim: -1);
                var batch_inds = torch.arange(B, device: device);

                low_res_masks = low_res_multimasks.index_select(0, batch_inds)
                    .index_select(1, best_iou_inds)
                    .unsqueeze(1);

                high_res_masks = high_res_multimasks.index_select(0, batch_inds)
                    .index_select(1, best_iou_inds)
                    .unsqueeze(1);

                if (sam_output_tokens.size(1) > 1)
                {
                    sam_output_token = sam_output_tokens.index_select(0, batch_inds)
                        .index_select(1, best_iou_inds);
                }
            }
            else
            {
                low_res_masks = low_res_multimasks;
                high_res_masks = high_res_multimasks;
            }

            // Extract object pointer
            var obj_ptr = obj_ptr_proj.forward(sam_output_token);

            if (pred_obj_scores)
            {
                Tensor lambda_is_obj_appearing;
                if (soft_no_obj_ptr)
                {
                    lambda_is_obj_appearing = torch.sigmoid(object_score_logits);
                }
                else
                {
                    lambda_is_obj_appearing = (object_score_logits > 0).to(ScalarType.Float32);
                }

                if (fixed_no_obj_ptr && no_obj_ptr is not null)
                {
                    obj_ptr = lambda_is_obj_appearing * obj_ptr;
                }

                obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * no_obj_ptr;
            }

            return (low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits);
        }

        public void set_image(Tensor image, Device device, ScalarType dtype)
        {
            using var _ = no_grad();
            //(Device device, ScalarType dtype) = Common.GetDeviceAndScaleType(this);
            if (image.shape.Length == 3)
            {
              //  image = image.unsqueeze(0);
            }
            else if (image.shape.Length > 4 || image.shape.Length < 3)
            {
                throw new ArgumentException("Image tensor's shape must be 3 or 4");
            }
            //long start = DateTime.Now.Ticks;
            //var image_embeddings = this.image_encoder.forward(image.to(dtype, device));
            var backbone_out = forward_image(image.to(dtype, device));
            //if (backbone_out.TryGetValue("vision_features", out var vfs) && vfs is Tensor vf)
            //{
            //    this.image_embeddings = vf;
            //}
            //var data = _prepare_backbone_features(backbone_out);

            //long mid = DateTime.Now.Ticks;
            //long MidGIelapsedMs = (mid - start) / TimeSpan.TicksPerMillisecond;
        }
        //public Tensor image_embeddings;
        /// </summary>
        /// <param name="batched_input">Batched inputs for SAM</param>
        /// <param name="multimask_output">Whether the model should predict multiple disambiguating masks, or return a single mask.</param>
        /// <returns></returns>
        int mask_threshold = 0;
        public BatchedOutput forward(BatchedInput image_record, bool multimask_output, bool return_logits = false)
        {
            //using var _ = NewDisposeScope();
            //(Device device, ScalarType dtype) = Common.GetDeviceAndScaleType(this);
            //if (this.image_embeddings is null)
            //{
            //    throw new NullReferenceException("Image Embeddings is null, please use SetImage before forward.");
            //}

            //(Tensor, Tensor)? points = image_record.Point_coords is not null ? (image_record.Point_coords, image_record.Point_labels) : null;
            //(Tensor sparse_embeddings, Tensor dense_embeddings) = sam2Sharp_prompt_encoder.forward(points: points, null);
            //(Tensor low_res_masks, Tensor iou_predictions) = this.sam2Sharp_mask_decoder.forward(image_embeddings: this.image_embeddings, image_pe: this.sam2Sharp_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings: sparse_embeddings, dense_prompt_embeddings: dense_embeddings, multimask_output: multimask_output);
            //Tensor masks = this.postprocess_masks(low_res_masks, /*input_size: image_record.Input_size,*/ orig_hw: image_record.orig_hw);
            //if (!return_logits)
            //{
            //    masks = masks > 0;
            //}

            //return new BatchedOutput
            //{
            //    Masks = masks.MoveToOuterDisposeScope(),
            //    Iou_predictions = iou_predictions.MoveToOuterDisposeScope(),
            //    Low_res_logits = low_res_masks.MoveToOuterDisposeScope()
            //};
            return null;
        }

        //public new BatchedOutput forward2(BatchedInput image_record, bool multimask_output, bool return_logits = false)
        //{
        //    using var _ = NewDisposeScope();
        //    (Device device, ScalarType dtype) = Common.GetDeviceAndScaleType(this);
        //    if (this.image_embeddings is null)
        //    {
        //        throw new NullReferenceException("Image Embeddings is null, please use SetImage before forward.");
        //    }
        //    image_record.to(device, dtype);
        //    (Tensor, Tensor)? points = image_record.Point_coords is not null ? (image_record.Point_coords, image_record.Point_labels) : null;
        //    (Tensor sparse_embeddings, Tensor dense_embeddings) = this.sam_prompt_encoder.forward(points: points,null, masks: image_record.Mask_inputs);
        //    List<Tensor> highResFeatures = null;
        //    //if (current_vision_feats.Count > 1 && feat_sizes.Count > 1)
        //    //{
        //    //    highResFeatures = new List<Tensor>();
        //    //    for (int i = 0; i < current_vision_feats.Count - 1; i++)
        //    //    {
        //    //        var feat = current_vision_feats[i];
        //    //        var (h, w) = feat_sizes[i];
        //    //        var reshapedFeat = feat.permute(1, 2, 0)
        //    //                               .view(feat.size(1), feat.size(2), h, w);
        //    //        highResFeatures.Add(reshapedFeat);
        //    //    }
        //    //}
        //    (Tensor low_res_masks, Tensor iou_predictions, Tensor mask_tokens_out, Tensor object_score_logits) = this.sam_mask_decoder.forward((image_embeddings: this.image_embeddings, image_pe: this.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings: sparse_embeddings, dense_prompt_embeddings: dense_embeddings, multimask_output: multimask_output, repeat_image: false, high_res_features: null));

        //    Tensor masks = this.postprocess_masks(low_res_masks, /*input_size: image_record.Input_size, */orig_hw: image_record.orig_hw);
        //    if (!return_logits)
        //    {
        //        masks = masks > this.mask_threshold;
        //    }

        //    return new BatchedOutput
        //    {
        //        Masks = masks.MoveToOuterDisposeScope(),
        //        Iou_predictions = iou_predictions.MoveToOuterDisposeScope(),
        //        Low_res_logits = low_res_masks.MoveToOuterDisposeScope()
        //    };
        //}
        /// <summary>
        /// Remove padding and upscale masks to the original image size.
        /// </summary>
        /// <param name="masks"> Batched masks from the mask_decoder, in BxCxHxW format.</param>
        /// <param name="input_size">The size of the image input to the model, in (H, W) format.Used to remove padding.</param>
        /// <param name="original_size">The original size of the image before resizing for input to the model, in (H, W) format.</param>
        /// <returns>Batched masks in BxCxHxW format, where (H, W) is given by original_size.</returns>
        public Tensor postprocess_masks(Tensor masks, /*long[] input_size,*/ (int h, int w) orig_hw)
        {
            torch.nn.functional.interpolate(masks,
                            size: [orig_hw.h, orig_hw.w],
                            mode: InterpolationMode.Bilinear,
                            align_corners: false);
            masks = torch.nn.functional.interpolate(masks, new long[] { this.image_size, this.image_size }, mode: InterpolationMode.Bilinear, align_corners: false);
            masks = masks[TensorIndex.Ellipsis, ..orig_hw.h, ..orig_hw.w];
            masks = torch.nn.functional.interpolate(masks, [orig_hw.h, orig_hw.w], mode: InterpolationMode.Bilinear, align_corners: false);
            return masks;
        }

   

        //public class SAM2VideoTracker : Module
        //{
        //    #region 模型配置参数（对应原PyTorch类的实例变量）
        //    // 编码器与解码器

        //    // 嵌入与投影层
        //private readonly Parameter no_mem_embed;
        //private readonly Parameter no_mem_pos_enc;
        //private readonly Parameter no_obj_embed_spatial;
        //private readonly Module obj_ptr_tpos_proj;
        //private readonly Tensor maskmem_tpos_enc;

        //    // 配置参数
        //public bool UseHighResFeaturesInSam { get; set; } = true;
        //public int num_feature_levels { get; set; }
        //public int num_maskmem { get; set; }
        //public int hidden_dim { get; set; }
        //public int mem_dim { get; set; }
        //public int max_cond_frames_in_attn { get; set; }
        //public int memory_temporal_stride_for_eval { get; set; }
        //public bool use_obj_ptrs_in_encoder { get; set; }
        //public int max_obj_ptrs_in_encoder { get; set; }
        //public bool only_obj_ptrs_in_the_past_for_eval { get; set; }
        //public bool use_signed_tpos_enc_to_obj_ptrs { get; set; }
        //public bool add_tpos_enc_to_obj_ptrs { get; set; }
        //public bool proj_tpos_enc_in_obj_ptrs { get; set; }
        //public bool directly_add_no_mem_embed { get; set; }
        //public bool non_overlap_masks_for_mem_enc { get; set; }
        //public bool binarize_mask_from_pts_for_mem_enc { get; set; }
        //public double sigmoid_scale_for_mem_enc { get; set; }
        //public double sigmoid_bias_for_mem_enc { get; set; }
        //public bool use_mask_input_as_output_without_sam { get; set; }
        //public bool multimask_output_in_sam { get; set; }
        //public bool multimask_output_for_tracking { get; set; }
        //public int multimask_min_pt_num { get; set; }
        //public int multimask_max_pt_num { get; set; }
        public bool Training { get; set; } = false;

        //    #endregion

        //    #region 构造函数（初始化模型组件与配置）
        //    public SAM2VideoTracker(
        //        Module imageEncoder,
        //        Module samMaskDecoder,
        //        Module memoryAttention,
        //        Module memory_encoder,
        //        Parameter noMemEmbed,
        //        Parameter noMemPosEnc,
        //        Parameter noObjEmbedSpatial = null,
        //        Module objPtrTposProj = null,
        //        Tensor maskmemTposEnc = null,
        //        string name = "SAM2VideoTracker") : base(name)
        //    {
        //        // 初始化模块组件
        //        image_encoder = imageEncoder;
        //        sam_mask_decoder = samMaskDecoder;
        //        memory_attention = memoryAttention;
        //        _memory_encoder = memory_encoder;
        //        no_mem_embed = noMemEmbed;
        //        no_mem_pos_enc = noMemPosEnc;
        //        no_obj_embed_spatial = noObjEmbedSpatial;
        //        obj_ptr_tpos_proj = objPtrTposProj;
        //        maskmem_tpos_enc = maskmemTposEnc;

        //        // 注册子模块与参数（TorchSharp必需）
        //        RegisterComponent("imageEncoder", image_encoder);
        //        RegisterComponent("samMaskDecoder", sam_mask_decoder);
        //        RegisterComponent("memoryAttention", memory_attention);
        //        RegisterComponent("memory_encoder", _memory_encoder);
        //        if (obj_ptr_tpos_proj != null) RegisterComponent("objPtrTposProj", obj_ptr_tpos_proj);
        //        RegisterParameter("noMemEmbed", no_mem_embed);
        //        RegisterParameter("noMemPosEnc", no_mem_pos_enc);
        //        if (no_obj_embed_spatial != null) RegisterParameter("noObjEmbedSpatial", no_obj_embed_spatial);

        //        // 默认配置初始化（可根据业务需求修改）
        //        UseHighResFeaturesInSam = false;
        //        num_feature_levels = 4;
        //        num_maskmem = 10;
        //        hidden_dim = 256;
        //        mem_dim = 256;
        //        max_cond_frames_in_attn = 5;
        //        memory_temporal_stride_for_eval = 1;
        //        use_obj_ptrs_in_encoder = true;
        //        max_obj_ptrs_in_encoder = 10;
        //        only_obj_ptrs_in_the_past_for_eval = true;
        //        use_signed_tpos_enc_to_obj_ptrs = false;
        //        add_tpos_enc_to_obj_ptrs = true;
        //        proj_tpos_enc_in_obj_ptrs = false;
        //        directly_add_no_mem_embed = false;
        //        non_overlap_masks_for_mem_enc = true;
        //        binarize_mask_from_pts_for_mem_enc = true;
        //        sigmoid_scale_for_mem_enc = 1.0;
        //        sigmoid_bias_for_mem_enc = 0.0;
        //        use_mask_input_as_output_without_sam = false;
        //        multimask_output_in_sam = true;
        //        multimask_output_for_tracking = true;
        //        multimask_min_pt_num = 1;
        //        multimask_max_pt_num = 10;
        //        Training = false;
        //    }
        //    #endregion

        //public void set_image(Tensor image, Device device, ScalarType dtype)
        //{
        //    using var _ = no_grad();
        //    //(Device device, ScalarType dtype) = Common.GetDeviceAndScaleType(this);
        //    if (image.shape.Length == 3)
        //    {
        //        image = image.unsqueeze(0);
        //    }
        //    else if (image.shape.Length > 4 || image.shape.Length < 3)
        //    {
        //        throw new ArgumentException("Image tensor's shape must be 3 or 4");
        //    }
        //    //long start = DateTime.Now.Ticks;
        //    this.image_embeddings = this.image_encoder.forward(this.preprocess(image).to(dtype, device));
        //    //long mid = DateTime.Now.Ticks;
        //    //long MidGIelapsedMs = (mid - start) / TimeSpan.TicksPerMillisecond;
        //}



        #region 1. forward_image：获取图像批次特征（对应原PyTorch方法）

        ///// <summary>
        ///// Get the image feature on the input batch.
        ///// </summary>
        ///// <param name="imgBatch">输入图像批次张量，形状：[B, 3, H, W]</param>
        ///// <returns>图像编码器输出结果（字典格式）</returns>
        public Dictionary<string, object> forward_image(Tensor img_batch)
        {
            var backbone_out = image_encoder.forward(img_batch);
            if (use_high_res_features_in_sam)
            {
                var fpnFeatures = backbone_out["backbone_fpn"] as Tensor[];
                // 原命名：conv_s0/conv_s1
                fpnFeatures[0] = sam_mask_decoder.conv_s0.forward(fpnFeatures[0]);
                fpnFeatures[1] = sam_mask_decoder.conv_s1.forward(fpnFeatures[1]);
                backbone_out["backbone_fpn"] = fpnFeatures;
            }
            return backbone_out;
        }
        
        #endregion

        #region 2. _prepare_backbone_features：准备并展平视觉特征（对应原PyTorch方法）
        /// <summary>
        /// Prepare and flatten visual features.
        /// </summary>
        /// <param name="backbone_out">图像编码器输出结果</param>
        /// <returns>处理后的骨干网络输出、视觉特征、视觉位置嵌入、特征尺寸</returns>
        public (Dictionary<string, object> backbone_out, List<Tensor> visionFeats, List<Tensor> visionPosEmbeds, List<(long, long)> feat_sizes)_prepare_backbone_features(Dictionary<string, object> backbone_out)
        {
        //    using var scope = NewDisposeScope();

            // 1. 深拷贝输入字典（避免修改原数据）
            var backbone_outCopy = new Dictionary<string, object>(backbone_out);
            foreach(var p in backbone_out)
            {
                if (p.Value is Tensor t)
                {
                    backbone_outCopy[p.Key] = torch.tensor(0, device: device);
                    backbone_outCopy[p.Key] = t.clone();
                }
                else if (p.Value is Tensor[] ta)
                {
                    var taClone = new Tensor[ta.Length];
                    for (int i = 0; i < ta.Length; i++)
                    {
                        taClone[i] = ta[i].clone();
                    }
                    backbone_outCopy[p.Key] = taClone;
                }
            }

            // 2. 合法性校验
            if (!backbone_outCopy.ContainsKey("backbone_fpn") || !backbone_outCopy.ContainsKey("vision_pos_enc"))
                throw new KeyNotFoundException("backbone_out缺失backbone_fpn或vision_pos_enc键");

            var backboneFpn = backbone_outCopy["backbone_fpn"] as Tensor[];
            var visionPosEnc = backbone_outCopy["vision_pos_enc"] as Tensor[];
            //if (backboneFpn == null || visionPosEnc == null)
            //    throw new InvalidOperationException("backbone_fpn或vision_pos_enc格式无效，需为List<Tensor>");
            //if (backboneFpn.Length != visionPosEnc.Length)
            //    throw new ArgumentException("backbone_fpn与vision_pos_enc长度不一致");
            //if (backboneFpn.Length < num_feature_levels)
            //    throw new ArgumentException($"backbone_fpn长度不足，期望至少{num_feature_levels}层，实际{backboneFpn.Length}层");

            // 3. 提取指定数量的顶层特征（从后往前取）
            var featureMaps = backboneFpn?.Skip(backboneFpn.Length - num_feature_levels).ToList();
            var visionPosEmbeds = visionPosEnc?.Skip(visionPosEnc.Length - num_feature_levels).ToList();

            // 4. 提取特征尺寸（H, W）
            var feat_sizes = new List<(long, long)>();
            foreach (var posEmbed in visionPosEmbeds)
            {
                long h = posEmbed.shape[^2];
                long w = posEmbed.shape[^1];
                feat_sizes.Add((h, w));
            }

            // 5. 展平特征：NxCxHxW → HWxNxC
            var visionFeats = new List<Tensor>();
            foreach (var feat in featureMaps)
            {
                var flattenedFeat = feat.flatten(2).permute(2, 0, 1);
                visionFeats.Add(flattenedFeat);
            }

            // 6. 展平位置嵌入：NxCxHxW → HWxNxC
            var flattenedVisionPosEmbeds = new List<Tensor>();
            foreach (var posEmbed in visionPosEmbeds)
            {
                var flattenedPosEmbed = posEmbed.flatten(2).permute(2, 0, 1);
                flattenedVisionPosEmbeds.Add(flattenedPosEmbed);
            }

            // 7. 返回结果（释放临时张量，保留核心输出）
            return (
                backbone_outCopy/*.MoveToOuterDisposeScope(scope)*/,
                visionFeats/*.MoveToOuterDisposeScope(scope)*/,
                flattenedVisionPosEmbeds/*.MoveToOuterDisposeScope(scope)*/,
                feat_sizes
            );
        }
        #endregion

        #region 3. _prepare_memory_conditioned_features：融合当前帧视觉特征与历史内存
        /// <summary>
        /// Fuse the current frame's visual feature map with previous memory.
        /// </summary>
        /// <param name="frame_idx">当前帧索引</param>
        /// <param name="is_init_cond_frame">是否为初始条件帧</param>
        /// <param name="current_vision_feats">当前帧视觉特征</param>
        /// <param name="current_vision_pos_embeds">当前帧视觉位置嵌入</param>
        /// <param name="feat_sizes">特征尺寸列表</param>
        /// <param name="output_dict">输出字典（包含历史帧结果）</param>
        /// <param name="num_frames">总帧数</param>
        /// <param name="track_in_reverse">是否反向跟踪</param>
        /// <returns>融合内存后的像素特征，形状：[B, C, H, W]</returns>
        public Tensor _prepare_memory_conditioned_features(
            int frame_idx,
            bool is_init_cond_frame,
            Tensor[] current_vision_feats,
            Tensor[] current_vision_pos_embeds,
            List<(long, long)> feat_sizes,
            Dictionary<string, object> output_dict,
            int? num_frames,
            bool track_in_reverse = false)
        {
            using var scope = NewDisposeScope();

            // 1. 提取基础参数
            if (current_vision_feats == null || current_vision_feats.Length == 0 || current_vision_pos_embeds == null || current_vision_pos_embeds.Length == 0)
                throw new ArgumentNullException("当前帧视觉特征或位置嵌入不能为空");
            if (feat_sizes == null || feat_sizes.Count == 0)
                throw new ArgumentNullException("特征尺寸列表不能为空");

            var topVisionFeat = current_vision_feats[^1];
            long B = topVisionFeat.shape[1]; // 批次大小
            long C = hidden_dim; // 隐藏层维度
            (long H, long W) = feat_sizes[^1]; // 顶层特征尺寸（最低分辨率）
            Device device = topVisionFeat.device;

            // 2. 禁用内存时，直接返回顶层特征重塑结果
            if (num_maskmem == 0)
            {
                var pixFeat = topVisionFeat.permute(1, 2, 0).view(B, C, H, W);
                return pixFeat.MoveToOuterDisposeScope();
            }

            long num_obj_ptr_tokens = 0;
            int tposSignMul = track_in_reverse ? -1 : 1;
            List<Tensor> toCatMemory = new List<Tensor>();
            List<Tensor> toCatMemoryPosEmbed = new List<Tensor>();

            // 3. 非初始条件帧：融合历史内存
            if (!is_init_cond_frame)
            {
                // 3.1 校验条件帧输出
                if (!output_dict.ContainsKey("cond_frame_outputs") || !(output_dict["cond_frame_outputs"] is Dictionary<int, Dictionary<string, object>> condOutputs))
                    throw new KeyNotFoundException("output_dict缺失有效cond_frame_outputs");
                if (condOutputs.Count == 0)
                    throw new InvalidOperationException("cond_frame_outputs为空，无法进行内存融合");

                // 3.2 选择时间上最近的条件帧
                var (selectedCondOutputs, unselectedCondOutputs) = SelectClosestCondFrames(
                    frame_idx, condOutputs, max_cond_frames_in_attn);

                // 3.3 构建时间位置与历史输出的映射
                var tPosAndPrevs = new List<(int, Dictionary<string, object>)>();
                foreach (var kvp in selectedCondOutputs)
                {
                    tPosAndPrevs.Add((0, kvp.Value));
                }

                // 3.4 补充历史非条件帧内存
                int stride = Training ? 1 : memory_temporal_stride_for_eval;
                for (int tPos = 1; tPos < num_maskmem; tPos++)
                {
                    int tRel = num_maskmem - tPos;
                    int prevframe_idx = 0;

                    if (tRel == 1)
                    {
                        // 取紧邻的前一帧/后一帧（反向跟踪）
                        prevframe_idx = track_in_reverse ? frame_idx + tRel : frame_idx - tRel;
                    }
                    else
                    {
                        // 取间隔stride的历史帧
                        if (!track_in_reverse)
                        {
                            prevframe_idx = ((frame_idx - 2) / stride) * stride;
                            prevframe_idx = prevframe_idx - (tRel - 2) * stride;
                        }
                        else
                        {
                            prevframe_idx = -(-(frame_idx + 2) / stride) * stride;
                            prevframe_idx = prevframe_idx + (tRel - 2) * stride;
                        }
                    }

                    // 提取非条件帧输出（若不存在，尝试从未选中的条件帧中获取）
                    Dictionary<string, object> prevOut = null;
                    if (output_dict.ContainsKey("non_cond_frame_outputs") && output_dict["non_cond_frame_outputs"] is Dictionary<int, Dictionary<string, object>> nonCondOutputs)
                    {
                        nonCondOutputs.TryGetValue(prevframe_idx, out prevOut);
                    }
                    prevOut ??= unselectedCondOutputs.TryGetValue(prevframe_idx, out var unselectedOut) ? unselectedOut : null;

                    tPosAndPrevs.Add((tPos, prevOut));
                }

                // 3.5 拼接历史内存特征与位置嵌入
                foreach (var (tPos, prev) in tPosAndPrevs)
                {
                    if (prev == null || !prev.ContainsKey("maskmem_features") || !prev.ContainsKey("maskmem_pos_enc"))
                        continue;

                    // 内存特征迁移到当前设备并展平
                    var feats = (prev["maskmem_features"] as Tensor).to(device, non_blocking: true);
                    var flattenedFeats = feats.flatten(2).permute(2, 0, 1);
                    toCatMemory.Add(flattenedFeats);

                    // 内存位置嵌入处理（空间+时间）
                    var maskmemEnc = prev["maskmem_pos_enc"] as List<object>;
                    if (maskmemEnc == null || maskmemEnc.Count == 0)
                        continue;

                    var flattenedMaskmemEnc = (maskmemEnc[^1] as Tensor).to(device).flatten(2).permute(2, 0, 1);
                    if (maskmem_tpos_enc is not null && tPos < num_maskmem)
                    {
                        var tposEnc = maskmem_tpos_enc[num_maskmem - tPos - 1].to(device);
                        flattenedMaskmemEnc = flattenedMaskmemEnc + tposEnc;
                    }
                    toCatMemoryPosEmbed.Add(flattenedMaskmemEnc);
                }

                // 3.6 构建对象指针（若启用）
                if (use_obj_ptrs_in_encoder)
                {
                    int max_obj_ptrs_in_encoder = Math.Min(num_frames ?? int.MaxValue, this.max_obj_ptrs_in_encoder);
                    Dictionary<int, Dictionary<string, object>> ptrCondOutputs = new Dictionary<int, Dictionary<string, object>>();

                    // 筛选条件帧中的对象指针（评估模式下仅保留过去的帧）
                    if (!Training && only_obj_ptrs_in_the_past_for_eval)
                    {
                        foreach (var kvp in selectedCondOutputs)
                        {
                            bool isPastFrame = track_in_reverse ? kvp.Key >= frame_idx : kvp.Key <= frame_idx;
                            if (isPastFrame)
                                ptrCondOutputs.Add(kvp.Key, kvp.Value);
                        }
                    }
                    else
                    {
                        ptrCondOutputs = selectedCondOutputs;
                    }

                    // 构建时间位置与对象指针的映射
                    var posAndPtrs = new List<(int, Tensor)>();
                    foreach (var kvp in ptrCondOutputs)
                    {
                        int t = kvp.Key;
                        var outDict = kvp.Value;
                        if (!outDict.ContainsKey("obj_ptr"))
                            continue;

                        // 计算时间位置编码
                        int tPos = use_signed_tpos_enc_to_obj_ptrs
                            ? (frame_idx - t) * tposSignMul
                            : Math.Abs(frame_idx - t);

                        posAndPtrs.Add((tPos, outDict["obj_ptr"] as Tensor));
                    }

                    // 补充非条件帧的对象指针
                    for (int tDiff = 1; tDiff < max_obj_ptrs_in_encoder; tDiff++)
                    {
                        int t = track_in_reverse ? frame_idx + tDiff : frame_idx - tDiff;
                        if (t < 0 || (num_frames.HasValue && t >= num_frames.Value))
                            break;

                        // 提取非条件帧/未选中条件帧的对象指针
                        Dictionary<string, object> outDict = null;
                        if (output_dict.ContainsKey("non_cond_frame_outputs") && output_dict["non_cond_frame_outputs"] is Dictionary<int, Dictionary<string, object>> nonCondOutputs)
                        {
                            nonCondOutputs.TryGetValue(t, out outDict);
                        }
                        outDict ??= unselectedCondOutputs.TryGetValue(t, out var unselectedOut) ? unselectedOut : null;

                        if (outDict != null && outDict.ContainsKey("obj_ptr"))
                        {
                            posAndPtrs.Add((tDiff, outDict["obj_ptr"] as Tensor));
                        }
                    }

                    // 拼接对象指针与位置嵌入
                    if (posAndPtrs.Count > 0)
                    {
                        var posList = posAndPtrs.Select(p => p.Item1).ToList();
                        var ptrsList = posAndPtrs.Select(p => p.Item2).ToList();

                        // 堆叠对象指针：[ptr_seq_len, B, C]
                        var objPtrs = stack(ptrsList.ToArray(), dim: 0);

                        // 生成时间位置嵌入
                        Tensor objPos = null;
                        if (add_tpos_enc_to_obj_ptrs)
                        {
                            int tDiffMax = max_obj_ptrs_in_encoder - 1;
                            int tposDim = proj_tpos_enc_in_obj_ptrs ? (int)C : mem_dim;

                            // 1D正弦位置编码
                            var objPosTensor = tensor(posList.ToArray(), device: device).to(ScalarType.Float32);
                            objPos = Get1dSinePe(objPosTensor / (float)tDiffMax, tposDim);
                            objPos = obj_ptr_tpos_proj.forward(objPos);
                            objPos = objPos.unsqueeze(1).expand(-1, B, mem_dim);
                        }
                        else
                        {
                            objPos = objPtrs.new_zeros(posList.Count, B, mem_dim);
                        }

                        // 处理维度不匹配（拆分对象指针）
                        if (mem_dim < C)
                        {
                            long splitNum = C / mem_dim;
                            objPtrs = objPtrs.reshape(-1, B, splitNum, mem_dim)
                                              .permute(0, 2, 1, 3)
                                              .flatten(0, 1);
                            objPos = objPos.repeat_interleave((long)splitNum, dim: 0);
                        }

                        // 添加到内存列表
                        toCatMemory.Add(objPtrs);
                        toCatMemoryPosEmbed.Add(objPos);
                        num_obj_ptr_tokens = objPtrs.shape[0];
                    }
                }
            }
            // 4. 初始条件帧：使用无内存嵌入（无历史内存）
            else
            {
                Tensor pixFeatWithMem2, pixFeat;
                if (directly_add_no_mem_embed)
                {
                    // 直接添加无内存嵌入，无需通过Transformer编码器
                    pixFeatWithMem2 = current_vision_feats[^1] + no_mem_embed;
                    pixFeat = pixFeatWithMem2.permute(1, 2, 0).view(B, C, H, W);
                    return pixFeat.MoveToOuterDisposeScope();
                }

                // 使用虚拟令牌避免空内存输入
                toCatMemory.Add(no_mem_embed.expand(1, B, mem_dim));
                toCatMemoryPosEmbed.Add(no_mem_pos_enc.expand(1, B, mem_dim));
            }

            // 5. 拼接内存并通过注意力层融合
            if (toCatMemory.Count == 0 || toCatMemoryPosEmbed.Count == 0)
                throw new InvalidOperationException("内存列表为空，无法进行特征融合");

            var memory = cat(toCatMemory.ToArray(), dim: 0);
            var memoryPosEmbed = cat(toCatMemoryPosEmbed.ToArray(), dim: 0);

            // 内存注意力前向传播
            var pixFeatWithMem = memory_attention.forward(
                curr: current_vision_feats[0],
                curr_pos: current_vision_pos_embeds[0],
                memory: memory,
                memory_pos: memoryPosEmbed,
                num_obj_ptr_tokens: (int)num_obj_ptr_tokens) as Tensor;

            if (pixFeatWithMem is null)
                throw new InvalidOperationException("内存注意力层输出格式无效，需返回Tensor");

            // 6. 重塑输出：(HW)BC → BCHW
            var finalPixFeat = pixFeatWithMem.permute(1, 2, 0).view(B, C, H, W);
            return finalPixFeat.MoveToOuterDisposeScope();
        }
        #endregion

        #region 4. _encode_new_memory：将当前帧与预测结果编码为内存特征
        /// <summary>
        /// Encode the current image and its prediction into a memory feature.
        /// </summary>
        /// <param name="current_vision_feats">当前帧视觉特征</param>
        /// <param name="feat_sizes">特征尺寸列表</param>
        /// <param name="predMasksHighRes">高分辨率预测掩码</param>
        /// <param name="objectScoreLogits">对象得分logits</param>
        /// <param name="isMaskFromPts">掩码是否来自点输入</param>
        /// <returns>内存特征与内存位置嵌入</returns>
        public (Tensor maskmemFeatures, Tensor maskmemPosEnc) EncodeNewMemory(
            List<Tensor> current_vision_feats,
            List<(long, long)> feat_sizes,
            Tensor predMasksHighRes,
            Tensor objectScoreLogits,
            bool isMaskFromPts)
        {
            using var scope = NewDisposeScope();

            // 1. 提取基础参数
            if (current_vision_feats == null || current_vision_feats.Count == 0 || feat_sizes == null || feat_sizes.Count == 0)
                throw new ArgumentNullException("当前帧视觉特征或特征尺寸列表不能为空");

            var topVisionFeat = current_vision_feats[^1];
            long B = topVisionFeat.shape[1];
            long C = hidden_dim;
            (long H, long W) = feat_sizes[^1];
            Device device = topVisionFeat.device;

            // 2. 重塑顶层视觉特征：(HW)BC → BCHW
            var pixFeat = topVisionFeat.permute(1, 2, 0).view(B, C, H, W);

            // 3. 非重叠掩码约束（评估模式下启用）
            if (non_overlap_masks_for_mem_enc && !Training)
            {
                predMasksHighRes = ApplyNonOverlappingConstraints(predMasksHighRes);
            }

            // 4. 掩码预处理（二值化或Sigmoid）
            Tensor maskForMem;
            bool binarize = binarize_mask_from_pts_for_mem_enc && isMaskFromPts;
            if (binarize && !Training)
            {
                maskForMem = (predMasksHighRes > 0).to(ScalarType.Float32);
            }
            else
            {
                maskForMem = sigmoid(predMasksHighRes);
            }

            // 5. 应用Sigmoid缩放与偏移
            if (sigmoid_scale_for_mem_enc != 1.0)
            {
                maskForMem = maskForMem * (float)sigmoid_scale_for_mem_enc;
            }
            if (sigmoid_bias_for_mem_enc != 0.0)
            {
                maskForMem = maskForMem + (float)sigmoid_bias_for_mem_enc;
            }

            // 6. 内存编码器前向传播
            var maskmemOut = memory_encoder.forward(
                pixFeat,
                maskForMem
                /*skip_mask_sigmoid: true*/) as Dictionary<string, Tensor>;

            if (maskmemOut == null || !maskmemOut.ContainsKey("vision_features") || !maskmemOut.ContainsKey("vision_pos_enc"))
                throw new InvalidOperationException("内存编码器输出格式无效，缺失vision_features或vision_pos_enc");

            var maskmemFeatures = maskmemOut["vision_features"] as Tensor;
            var maskmemPosEnc = maskmemOut["vision_pos_enc"] as Tensor;

            if (maskmemFeatures is null || maskmemPosEnc is null)
                throw new InvalidOperationException("内存编码器输出的特征或位置嵌入格式无效");

            // 7. 添加无对象嵌入（若启用）
            if (no_obj_embed_spatial is not null)
            {
                var isObjAppearing = (objectScoreLogits > 0).to(ScalarType.Float32);
                var noObjEmbedExpanded = no_obj_embed_spatial.unsqueeze(-1).unsqueeze(-1)
                                                            .expand(maskmemFeatures.shape);
                var noObjMask = (1 - isObjAppearing.unsqueeze(-1).unsqueeze(-1))
                                 .expand(maskmemFeatures.shape);

                maskmemFeatures = maskmemFeatures + (noObjMask * noObjEmbedExpanded);
            }

            // 8. 返回结果
            return (
                maskmemFeatures!.MoveToOuterDisposeScope(),
                maskmemPosEnc!.MoveToOuterDisposeScope()
            );
        }
        #endregion

        #region 5. _track_step：跟踪单步处理（对应原PyTorch方法）
        /// <summary>
        /// 单步跟踪核心逻辑
        /// </summary>
        /// <returns>当前帧输出、SAM输出、高分辨率特征、像素特征</returns>
        public (Dictionary<string, object> currentOut, (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) samOutputs, List<Tensor> highResFeatures, Tensor pixFeat)
            TrackStepInternal(
                int frame_idx,
                bool is_init_cond_frame,
                List<Tensor> current_vision_feats,
                List<Tensor> current_vision_pos_embeds,
                List<(long, long)> feat_sizes,
                (Tensor, Tensor) pointInputs,
                Tensor maskInputs,
                Dictionary<string, object> output_dict,
                int? num_frames,
                bool track_in_reverse,
                Tensor prevSamMaskLogits)
        {
            using var scope = NewDisposeScope();

            // 1. 初始化当前帧输出
            var currentOut = new Dictionary<string, object>
            {
                ["point_inputs"] = pointInputs,
                ["mask_inputs"] = maskInputs
            };

            // 2. 提取高分辨率特征
            List<Tensor> highResFeatures = null;
            if (current_vision_feats.Count > 1 && feat_sizes.Count > 1)
            {
                highResFeatures = new List<Tensor>();
                for (int i = 0; i < current_vision_feats.Count - 1; i++)
                {
                    var feat = current_vision_feats[i];
                    var (h, w) = feat_sizes[i];
                    var reshapedFeat = feat.permute(1, 2, 0)
                                           .view(feat.size(1), feat.size(2), h, w);
                    highResFeatures.Add(reshapedFeat);
                }
            }

            // 3. 直接使用掩码输入作为输出（若启用）
            (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) samOutputs = default;
            Tensor pixFeat = null;
            if (maskInputs is not null && use_mask_input_as_output_without_sam)
            {
                pixFeat = current_vision_feats[^1].permute(1, 2, 0).view(-1, hidden_dim, feat_sizes[^1].Item1, feat_sizes[^1].Item2);
                samOutputs = _use_mask_as_output(pixFeat, highResFeatures, maskInputs);
            }
            else
            {
                // 3.1 融合视觉特征与历史内存
                pixFeat = _prepare_memory_conditioned_features(
                    frame_idx: frame_idx,
                    is_init_cond_frame: is_init_cond_frame,
                    current_vision_feats: current_vision_feats.Skip(current_vision_feats.Count - 1).ToArray(),
                    current_vision_pos_embeds: current_vision_pos_embeds.Skip(current_vision_pos_embeds.Count - 1).ToArray(),
                    feat_sizes: feat_sizes.Skip(feat_sizes.Count - 1).ToList(),
                    output_dict: output_dict,
                    num_frames: num_frames,
                    track_in_reverse: track_in_reverse);

                // 3.2 预处理前序SAM掩码logits
                if (prevSamMaskLogits is not null)
                {
                    if (pointInputs.Item1 is null || maskInputs is not null)
                        throw new InvalidOperationException("前序SAM掩码logits仅支持点输入且无掩码输入的场景");
                    maskInputs = prevSamMaskLogits;
                }

                // 3.3 判断是否使用多掩码输出
                bool multimaskOutput = UseMultimask(is_init_cond_frame, pointInputs);

                // 3.4 SAM头前向传播
                samOutputs = _forward_sam_heads(
                    backbone_features: pixFeat,
                    point_inputs: pointInputs,
                    mask_inputs: maskInputs,
                    high_res_features: highResFeatures,
                    multimask_output: multimaskOutput);
            }

            // 4. 返回结果
            return (
                currentOut,
                samOutputs,
                highResFeatures,
                pixFeat.MoveToOuterDisposeScope()
            );
        }
        #endregion

        #region 6. _encode_memory_in_output：将内存编码到输出结果中（对应原PyTorch方法）
        /// <summary>
        /// 将当前帧预测结果编码为内存并更新输出字典
        /// </summary>
        public void EncodeMemoryInOutput(
            List<Tensor> current_vision_feats,
            List<(long, long)> feat_sizes,
            (Tensor, Tensor) pointInputs,
            bool runMemEncoder,
            Tensor highResMasks,
            Tensor objectScoreLogits,
            Dictionary<string, object> currentOut)
        {
            using var scope = NewDisposeScope();

            if (runMemEncoder && num_maskmem > 0)
            {
                var highResMasksForMemEnc = highResMasks;
                var (maskmemFeatures, maskmemPosEnc) = EncodeNewMemory(
                    current_vision_feats: current_vision_feats,
                    feat_sizes: feat_sizes,
                    predMasksHighRes: highResMasksForMemEnc,
                    objectScoreLogits: objectScoreLogits,
                    isMaskFromPts: (pointInputs.Item1 is not null));

                currentOut["maskmem_features"] = maskmemFeatures;
                currentOut["maskmem_pos_enc"] = maskmemPosEnc;
            }
            else
            {
                currentOut["maskmem_features"] = null;
                currentOut["maskmem_pos_enc"] = null;
            }
        }
        #endregion

        #region 7. track_step：公开跟踪单步方法（对应原PyTorch方法）
        /// <summary>
        /// 公开单步跟踪方法，处理当前帧并返回输出结果
        /// </summary>
        /// <returns>当前帧完整输出结果</returns>
        public Dictionary<string, object> TrackStep(
            int frame_idx,
            bool is_init_cond_frame,
            List<Tensor> current_vision_feats,
            List<Tensor> current_vision_pos_embeds,
            List<(long, long)> feat_sizes,
            (Tensor, Tensor) pointInputs,
            Tensor maskInputs,
            Dictionary<string, object> output_dict,
            int? num_frames,
            bool track_in_reverse = false,
            bool runMemEncoder = true,
            Tensor prevSamMaskLogits = null)
        {
            using var scope = NewDisposeScope();

            // 1. 内部单步跟踪处理
            var (currentOut, samOutputs, _, _) = TrackStepInternal(
                frame_idx,
                is_init_cond_frame,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                pointInputs,
                maskInputs,
                output_dict,
                num_frames,
                track_in_reverse,
                prevSamMaskLogits);

            // 2. 解析SAM输出
            var (_, _, _, lowResMasks, highResMasks, objPtr, objectScoreLogits) = samOutputs;

            // 3. 更新当前帧输出
            currentOut["pred_masks"] = lowResMasks;
            currentOut["pred_masks_high_res"] = highResMasks;
            currentOut["obj_ptr"] = objPtr;
            if (!Training)
            {
                currentOut["object_score_logits"] = objectScoreLogits;
            }

            // 4. 编码内存特征
            EncodeMemoryInOutput(
                current_vision_feats,
                feat_sizes,
                pointInputs,
                runMemEncoder,
                highResMasks,
                objectScoreLogits,
                currentOut);

            // 5. 返回结果
            return currentOut;
        }
        #endregion

        #region 8. 辅助方法：对应原PyTorch工具方法
        /// <summary>
        /// Whether to use multimask output in the SAM head.
        /// </summary>
        public bool UseMultimask(bool is_init_cond_frame, (Tensor, Tensor) pointInputs)
        {
            int numPts = pointInputs.Item1 is null? 0: (int)pointInputs.Item2.shape[1];

            return multimask_output_in_sam
                   && (is_init_cond_frame || multimask_output_for_tracking)
                   && (multimask_min_pt_num <= numPts && numPts <= multimask_max_pt_num);
        }

        /// <summary>
        /// Apply non-overlapping constraints to the object scores in pred_masks.
        /// </summary>
        public Tensor ApplyNonOverlappingConstraints(Tensor predMasks)
        {
            using var scope = NewDisposeScope();

            long batchSize = predMasks.shape[0];
            if (batchSize == 1)
                return predMasks.MoveToOuterDisposeScope();

            Device device = predMasks.device;

            // 计算每个空间位置得分最高的对象索引
            var maxObjInds = argmax(predMasks, dim: 0, keepdim: true);

            // 构建批次对象索引
            var batchObjInds = arange(batchSize, device: device)
                .unsqueeze(1).unsqueeze(2).unsqueeze(3);

            // 生成保留掩码
            var keep = maxObjInds == batchObjInds;

            // 抑制重叠区域得分（低于-10.0，对应Sigmoid后接近0）
            var clampedMasks = clamp(predMasks, max: -10.0f);
            var result = where(keep, predMasks, clampedMasks);

            return result.MoveToOuterDisposeScope();
        }
        #endregion

        #region 9. 占位辅助方法：需根据原PyTorch逻辑补充实现
        /// <summary>
        /// 选择时间上最近的条件帧（占位方法，需补充具体实现）
        /// </summary>
        private (Dictionary<int, Dictionary<string, object>> selected, Dictionary<int, Dictionary<string, object>> unselected)
            SelectClosestCondFrames(int frame_idx, Dictionary<int, Dictionary<string, object>> condOutputs, int maxCondFrames)
        {
            // 此处需根据原PyTorch逻辑实现条件帧筛选逻辑
            var selected = new Dictionary<int, Dictionary<string, object>>();
            var unselected = new Dictionary<int, Dictionary<string, object>>();

            // 临时实现：取前maxCondFrames个帧
            int count = 0;
            foreach (var kvp in condOutputs.OrderBy(k => Math.Abs(k.Key - frame_idx)))
            {
                if (count < maxCondFrames)
                {
                    selected.Add(kvp.Key, kvp.Value);
                    count++;
                }
                else
                {
                    unselected.Add(kvp.Key, kvp.Value);
                }
            }

            return (selected, unselected);
        }

        /// <summary>
        /// 生成1D正弦位置编码（占位方法，需补充具体实现）
        /// </summary>
        private Tensor Get1dSinePe(Tensor pos, int dim)
        {
            // 此处需根据原PyTorch的get_1d_sine_pe逻辑实现
            return pos.unsqueeze(-1).expand(-1, dim);
        }

        /// <summary>
        /// Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        /// (same input and output shapes as in _forward_sam_heads above).
        /// </summary>
        private (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) _use_mask_as_output(Tensor backbone_features,List<Tensor> high_res_features,Tensor mask_inputs)
        {
            // 1. 原命名：out_scale, out_bias
            float out_scale = 20.0f;
            float out_bias = -10.0f;

            // 2. 原命名：mask_inputs_float
            var mask_inputs_float = mask_inputs.to(ScalarType.Float32);
            var high_res_masks = mask_inputs_float * out_scale + out_bias;

            // 3. 下采样低分辨率掩码（原命名：low_res_masks）
            long h = high_res_masks.size(-2) / 4;
            long w = high_res_masks.size(-1) / 4;
            var low_res_masks = functional.interpolate(
                high_res_masks,
                size: new long[] { h, w },
                align_corners: false,
                mode: InterpolationMode.Bilinear,
                antialias: true);

            // 4. 原命名：ious
            var ious = mask_inputs.new_ones(mask_inputs.size(0), 1).to(ScalarType.Float32);

            // 5. 原命名：obj_ptr
            Tensor obj_ptr;
            if (!use_obj_ptrs_in_encoder)
            {
                obj_ptr = torch.zeros(
                    new long[] { mask_inputs.size(0), hidden_dim },
                    device: mask_inputs.device);
            }
            else
            {
                // 原命名：_forward_sam_heads
                var (_, _, _, _, _, obj_ptr_from_sam, _) = _forward_sam_heads(
                    backbone_features: backbone_features,
                    mask_inputs: mask_downsample.forward(mask_inputs_float),
                    point_inputs: (null, null),
                    high_res_features: high_res_features);
                obj_ptr = obj_ptr_from_sam;
            }

            // 6. 原命名：is_obj_appearing, lambda_is_obj_appearing
            var is_obj_appearing = torch.any(mask_inputs.flatten(1).to(ScalarType.Float32) > 0.0f, dim: 1);
            is_obj_appearing = is_obj_appearing.unsqueeze(-1);
            var lambda_is_obj_appearing = is_obj_appearing.to(ScalarType.Float32);

            // 7. 原命名：object_score_logits
            var object_score_logits = out_scale * lambda_is_obj_appearing + out_bias;

            // 8. 原逻辑保留
            if (pred_obj_scores)
            {
                if (fixed_no_obj_ptr)
                {
                    obj_ptr = lambda_is_obj_appearing * obj_ptr;
                }
                obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * no_obj_ptr;
            }

            return (
                low_res_masks,
                high_res_masks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits
            );
        }



        /// <summary>
        /// Fuse the current frame's visual feature map with previous memory.
        /// </summary>
        public Tensor prepare_memory_conditioned_features(int frame_idx,bool is_init_cond_frame,List<Tensor> current_vision_feats,List<Tensor> current_vision_pos_embeds,List<(long, long)> feat_sizes,Dictionary<string, object> output_dict,int? num_frames,bool track_in_reverse = false) // tracking in reverse time order (for demo usage)
        {
            // 1. 获取基础维度信息
            long B = current_vision_feats[^1].size(1); // batch size on this frame
            int C = hidden_dim;
            (long H, long W) = feat_sizes[^1]; // top-level (lowest-resolution) feature size
            Device device = current_vision_feats[^1].device;

            // 2. 禁用内存融合（用于复现SAM图像推理）
            if (num_maskmem == 0)
            {
                var pix_feat = current_vision_feats[^1].permute(new long[] { 1, 2, 0 })
                    .view(new long[] { B, C, H, W });
                return pix_feat;
            }

            int num_obj_ptr_tokens = 0;
            int tpos_sign_mul = track_in_reverse ? -1 : 1;

            // 3. Step 1: 基于历史内存条件化当前帧视觉特征
            List<Tensor> to_cat_memory = new();
            List<Tensor> to_cat_memory_pos_embed = new();

            if (!is_init_cond_frame)
            {
                // 3.1 检索掩码内存骨干编码的历史内存
                var cond_outputs = (Dictionary<int, Dictionary<string, object>>)output_dict["cond_frame_outputs"];
                System.Diagnostics.Debug.Assert(cond_outputs.Count > 0);

                // 3.2 选择时间上最近的条件帧用于交叉注意力
                var (selected_cond_outputs, unselected_cond_outputs) = Sam2Utils.select_closest_cond_frames(frame_idx, cond_outputs, max_cond_frames_in_attn);

                // 3.3 初始化条件帧的时间位置和输出
                List<(int, Dictionary<string, object>)> t_pos_and_prevs = selected_cond_outputs
                    .Select(kv => (0, kv.Value))
                    .ToList();

                // 3.4 添加当前帧之前的 (num_maskmem - 1) 帧作为非条件内存
                int stride = training ? 1 : memory_temporal_stride_for_eval;
                for (int t_pos = 1; t_pos < num_maskmem; t_pos++)
                {
                    int t_rel = num_maskmem - t_pos; // 距离当前帧的帧数
                    int prev_frame_idx;

                    if (t_rel == 1)
                    {
                        // 取紧邻的前一帧/后一帧（反向追踪）
                        prev_frame_idx = track_in_reverse ? frame_idx + t_rel : frame_idx - t_rel;
                    }
                    else
                    {
                        // 取间隔stride的历史帧
                        if (!track_in_reverse)
                        {
                            prev_frame_idx = ((frame_idx - 2) / stride) * stride;
                            prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride;
                        }
                        else
                        {
                            prev_frame_idx = -(-(frame_idx + 2) / stride) * stride;
                            prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride;
                        }
                    }

                    // 3.5 获取非条件帧输出（无则取未选中的条件帧）
                    var non_cond_outputs = (Dictionary<int, Dictionary<string, object>>)output_dict["non_cond_frame_outputs"];
                    Dictionary<string, object> out_dict = null;
                    if (non_cond_outputs.ContainsKey(prev_frame_idx))
                    {
                        out_dict = non_cond_outputs[prev_frame_idx];
                    }
                    else if (unselected_cond_outputs.ContainsKey(prev_frame_idx))
                    {
                        out_dict = unselected_cond_outputs[prev_frame_idx];
                    }

                    t_pos_and_prevs.Add((t_pos, out_dict));
                }

                // 3.6 拼接内存特征和位置编码
                foreach (var (t_pos, prev) in t_pos_and_prevs)
                {
                    if (prev == null) continue; // 跳过填充帧

                    // 加载内存特征到目标设备（兼容CPU/GPU offload）
                    Tensor feats = ((Tensor)prev["maskmem_features"]).to(device, non_blocking: true);
                    to_cat_memory.Add(feats.flatten(2).permute(new long[] { 2, 0, 1 }));

                    // 空间位置编码 + 时间位置编码
                    Tensor maskmem_enc = ((List<Tensor>)prev["maskmem_pos_enc"])[^1].to(device);
                    maskmem_enc = maskmem_enc.flatten(2).permute(new long[] { 2, 0, 1 });
                    maskmem_enc = maskmem_enc + maskmem_tpos_enc[num_maskmem - t_pos - 1];
                    to_cat_memory_pos_embed.Add(maskmem_enc);
                }

                // 3.7 构建历史对象指针列表
                if (use_obj_ptrs_in_encoder)
                {
                    int max_obj_ptrs_in_encoder = Math.Min(num_frames ?? int.MaxValue, this.max_obj_ptrs_in_encoder);
                    Dictionary<int, Dictionary<string, object>> ptr_cond_outputs;

                    // 评估阶段仅使用过去的对象指针
                    if (!training && only_obj_ptrs_in_the_past_for_eval)
                    {
                        ptr_cond_outputs = selected_cond_outputs
                            .Where(kv => track_in_reverse ? kv.Key >= frame_idx : kv.Key <= frame_idx)
                            .ToDictionary(kv => kv.Key, kv => kv.Value);
                    }
                    else
                    {
                        ptr_cond_outputs = selected_cond_outputs;
                    }

                    // 3.8 生成位置和指针的映射
                    List<(int, Tensor)> pos_and_ptrs = ptr_cond_outputs
                        .Select(kv =>
                        {
                            int t = kv.Key;
                            int pos = use_signed_tpos_enc_to_obj_ptrs
                                ? (frame_idx - t) * tpos_sign_mul
                                : Math.Abs(frame_idx - t);
                            Tensor ptr = (Tensor)kv.Value["obj_ptr"];
                            return (pos, ptr);
                        })
                        .ToList();

                    // 3.9 添加非条件帧的对象指针
                    for (int t_diff = 1; t_diff < max_obj_ptrs_in_encoder; t_diff++)
                    {
                        int t = track_in_reverse ? frame_idx + t_diff : frame_idx - t_diff;
                        if (t < 0 || (num_frames.HasValue && t >= num_frames.Value))
                            break;

                        Dictionary<string, object> out_dict = null;
                        var non_cond_outputs = (Dictionary<int, Dictionary<string, object>>)output_dict["non_cond_frame_outputs"];
                        if (non_cond_outputs.ContainsKey(t))
                        {
                            out_dict = non_cond_outputs[t];
                        }
                        else if (unselected_cond_outputs.ContainsKey(t))
                        {
                            out_dict = unselected_cond_outputs[t];
                        }

                        if (out_dict != null)
                        {
                            pos_and_ptrs.Add((t_diff, (Tensor)out_dict["obj_ptr"]));
                        }
                    }

                    // 3.10 处理对象指针（添加到内存）
                    if (pos_and_ptrs.Count > 0)
                    {
                        var pos_list = pos_and_ptrs.Select(x => x.Item1).ToList();
                        var ptrs_list = pos_and_ptrs.Select(x => x.Item2).ToList();

                        // 堆叠对象指针 [ptr_seq_len, B, C]
                        Tensor obj_ptrs = torch.stack(ptrs_list.ToArray(), dim: 0);

                        // 时间位置编码
                        Tensor obj_pos;
                        if (add_tpos_enc_to_obj_ptrs)
                        {
                            int t_diff_max = max_obj_ptrs_in_encoder - 1;
                            int tpos_dim = proj_tpos_enc_in_obj_ptrs ? C : mem_dim;

                            var obj_pos_tensor = torch.tensor(pos_list.ToArray(), device: device);
                            obj_pos = Sam2Utils.get_1d_sine_pe(obj_pos_tensor / (float)t_diff_max, dim: tpos_dim);
                            obj_pos = obj_ptr_tpos_proj.forward(obj_pos);
                            obj_pos = obj_pos.unsqueeze(1).expand(new long[] { -1, B, mem_dim });
                        }
                        else
                        {
                            obj_pos = obj_ptrs.new_zeros(new long[] { pos_list.Count, B, mem_dim });
                        }

                        // 拆分指针（mem_dim < C时）
                        if (mem_dim < C)
                        {
                            int split_dim = C / mem_dim;
                            obj_ptrs = obj_ptrs.reshape(new long[] { -1, B, split_dim, mem_dim });
                            obj_ptrs = obj_ptrs.permute(new long[] { 0, 2, 1, 3 }).flatten(0, 1);
                            obj_pos = obj_pos.repeat_interleave(split_dim, dim: 0);
                        }

                        to_cat_memory.Add(obj_ptrs);
                        to_cat_memory_pos_embed.Add(obj_pos);
                        num_obj_ptr_tokens = (int)obj_ptrs.shape[0];
                    }
                    else
                    {
                        num_obj_ptr_tokens = 0;
                    }
                }
            }
            else
            {
                // 3.11 初始条件帧：无历史内存
                if (directly_add_no_mem_embed)
                {
                    // 直接添加无内存嵌入
                    var pix_feat_with_mem2 = current_vision_feats[^1] + no_mem_embed;
                    pix_feat_with_mem2 = pix_feat_with_mem2.permute(new long[] { 1, 2, 0 })
                        .view(new long[] { B, C, H, W });
                    return pix_feat_with_mem2;
                }

                // 使用dummy token避免空内存输入
                to_cat_memory.Add(no_mem_embed.expand(new long[] { 1, B, mem_dim }));
                to_cat_memory_pos_embed.Add(no_mem_pos_enc.expand(new long[] { 1, B, mem_dim }));
            }

            // 4. Step 2: 拼接内存并通过Transformer编码器前向传播
            Tensor memory = torch.cat(to_cat_memory.ToArray(), dim: 0);
            Tensor memory_pos_embed = torch.cat(to_cat_memory_pos_embed.ToArray(), dim: 0);

            // 5. 内存注意力融合
            var pix_feat_with_mem = memory_attention.forward(
                curr: stack(current_vision_feats),
                curr_pos: stack(current_vision_pos_embeds),
                memory: memory,
                memory_pos: memory_pos_embed,
                num_obj_ptr_tokens: num_obj_ptr_tokens);

            // 6. 重塑输出 (HW)BC → BCHW
            pix_feat_with_mem = pix_feat_with_mem.permute(new long[] { 1, 2, 0 })
                .view(new long[] { B, C, H, W });

            return pix_feat_with_mem;
        }

        /// <summary>
        /// Encode the current image and its prediction into a memory feature.
        /// </summary>
        private (Tensor, Tensor) _encode_new_memory(
            Tensor current_vision_feats,
            List<(long, long)> feat_sizes,
            Tensor pred_masks_high_res,
            Tensor object_score_logits,
            bool is_mask_from_pts)
        {
            // 1. 获取基础维度信息
            long B = current_vision_feats[^1].size(1); // batch size on this frame
            int C = hidden_dim;
            (long H, long W) = feat_sizes[^1]; // top-level feature size

            // 2. 顶级特征重塑 (HW)BC → BCHW
            var pix_feat = current_vision_feats[^1].permute(new long[] { 1, 2, 0 })
                .view(new long[] { B, C, H, W });

            // 3. 非重叠掩码约束（仅评估阶段）
            if (non_overlap_masks_for_mem_enc && !training)
            {
                pred_masks_high_res = _apply_non_overlapping_constraints(pred_masks_high_res);
            }

            // 4. 处理掩码用于内存编码
            bool binarize = binarize_mask_from_pts_for_mem_enc && is_mask_from_pts;
            Tensor mask_for_mem;

            if (binarize && !training)
            {
                mask_for_mem = (pred_masks_high_res > 0).to(ScalarType.Float32);
            }
            else
            {
                // 对原始掩码logits应用sigmoid
                mask_for_mem = sigmoid(pred_masks_high_res);
            }

            // 5. 应用sigmoid缩放和偏移
            if (sigmoid_scale_for_mem_enc != 1.0f)
            {
                mask_for_mem = mask_for_mem * sigmoid_scale_for_mem_enc;
            }
            if (sigmoid_bias_for_mem_enc != 0.0f)
            {
                mask_for_mem = mask_for_mem + sigmoid_bias_for_mem_enc;
            }

            // 6. 内存编码器前向传播
            var maskmem_out = memory_encoder.forward(
                pix_feat,
                mask_for_mem,
                skip_mask_sigmoid: true); // sigmoid已提前应用

            Tensor maskmem_features = (Tensor)maskmem_out["vision_features"];
            Tensor maskmem_pos_enc = (Tensor)maskmem_out["vision_pos_enc"];

            // 7. 添加无对象嵌入（指示帧被遮挡/无对象）
            if (no_obj_embed_spatial is not null)
            {
                var is_obj_appearing = (object_score_logits > 0).to(ScalarType.Float32);
                var no_obj_embed_expanded = no_obj_embed_spatial
                    .unsqueeze(-1).unsqueeze(-1)
                    .expand(maskmem_features.shape);

                maskmem_features += (1 - is_obj_appearing.unsqueeze(-1).unsqueeze(-1)) * no_obj_embed_expanded;
            }

            return (maskmem_features, maskmem_pos_enc);
        }
        /// <summary>
        /// 单帧跟踪步骤核心逻辑
        /// </summary>
        private (Dictionary<string, object>, (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor), List<Tensor>, Tensor) _track_step(
            int frame_idx,
            bool is_init_cond_frame,
            List<Tensor> current_vision_feats,
            List<Tensor> current_vision_pos_embeds,
            List<(long, long)> feat_sizes,
            (Tensor,Tensor) point_inputs, // 兼容点输入的多类型结构
            Tensor mask_inputs,
            Dictionary<string, object> output_dict,
            int? num_frames,
            bool track_in_reverse,
            Tensor prev_sam_mask_logits)
        {
            // 1. 初始化当前帧输出字典
            var current_out = new Dictionary<string, object>
            {
                ["point_inputs"] = point_inputs,
                ["mask_inputs"] = mask_inputs
            };

            // 2. 处理高分辨率特征（SAM Head用）：(HW)BC → BCHW
            List<Tensor> high_res_features = null;
            if (current_vision_feats.Count > 1)
            {
                high_res_features = new List<Tensor>();
                for (int i = 0; i < current_vision_feats.Count - 1; i++)
                {
                    var x = current_vision_feats[i];
                    var s = feat_sizes[i];
                    // 维度变换：(HW)BC → BCHW
                    var feat = x.permute(new long[] { 1, 2, 0 })
                        .view(new long[] { x.size(1), x.size(2), s.Item1, s.Item2 });
                    high_res_features.Add(feat);
                }
            }

            // 3. 分支1：直接使用掩码输入作为输出（不调用SAM）
            Tensor pix_feat = null;
            (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) sam_outputs;

            if (mask_inputs is not null && use_mask_input_as_output_without_sam)
            {
                // 转换顶级特征：(HW)BC → BCHW
                pix_feat = current_vision_feats[^1].permute(new long[] { 1, 2, 0 });
                pix_feat = pix_feat.view(new long[] { -1, hidden_dim, feat_sizes[^1].Item1, feat_sizes[^1].Item2 });

                // 调用_use_mask_as_output方法
                sam_outputs = _use_mask_as_output(
                    pix_feat,
                    high_res_features,
                    mask_inputs);
            }
            else
            {
                // 4. 分支2：融合视觉特征与历史内存特征
                pix_feat = _prepare_memory_conditioned_features(
                    frame_idx: frame_idx,
                    is_init_cond_frame: is_init_cond_frame,
                    current_vision_feats: current_vision_feats.Skip(current_vision_feats.Count - 1).ToArray(),
                    current_vision_pos_embeds: current_vision_pos_embeds.Skip(current_vision_pos_embeds.Count - 1).ToArray(),
                    feat_sizes: feat_sizes.Skip(feat_sizes.Count - 1).ToList(),
                    output_dict: output_dict,
                    num_frames: num_frames,
                    track_in_reverse: track_in_reverse);

                // 5. 处理历史SAM掩码logits（演示场景下的交互输入）
                if (prev_sam_mask_logits is not null)
                {
                    System.Diagnostics.Debug.Assert(point_inputs.Item1 is not null && mask_inputs is null,
                        "prev_sam_mask_logits不为空时，point_inputs必须非空且mask_inputs必须为空");
                    mask_inputs = prev_sam_mask_logits;
                }

                // 6. 判断是否输出多掩码
                bool multimask_output = _use_multimask(is_init_cond_frame, point_inputs);

                // 7. 调用SAM Heads前向传播
                sam_outputs = _forward_sam_heads(
                    backbone_features: pix_feat,
                    point_inputs: point_inputs ,
                    mask_inputs: mask_inputs,
                    high_res_features: high_res_features,
                    multimask_output: multimask_output);
            }

            return (current_out, sam_outputs, high_res_features, pix_feat);
        }

        /// <summary>
        /// 将预测结果编码为内存特征并写入输出字典
        /// </summary>
        private void _encode_memory_in_output(
            Tensor current_vision_feats,
            List<(long, long)> feat_sizes,
            object point_inputs,
            bool run_mem_encoder,
            Tensor high_res_masks,
            Tensor object_score_logits,
            Dictionary<string, object> current_out)
        {
            if (run_mem_encoder && num_maskmem > 0)
            {
                // 1. 准备用于内存编码的高分辨率掩码
                var high_res_masks_for_mem_enc = high_res_masks;

                // 2. 编码新内存特征
                var (maskmem_features, maskmem_pos_enc) = _encode_new_memory(
                    current_vision_feats: current_vision_feats,
                    feat_sizes: feat_sizes,
                    pred_masks_high_res: high_res_masks_for_mem_enc,
                    object_score_logits: object_score_logits,
                    is_mask_from_pts: (point_inputs is not null));

                // 3. 写入输出字典
                current_out["maskmem_features"] = maskmem_features;
                current_out["maskmem_pos_enc"] = maskmem_pos_enc;
            }
            else
            {
                // 4. 禁用内存编码器时写入空值
                current_out["maskmem_features"] = null;
                current_out["maskmem_pos_enc"] = null;
            }
        }

        /// <summary>
        /// </summary>
        public Dictionary<string, object> track_step(
            int frame_idx,
            bool is_init_cond_frame,
            List<Tensor> current_vision_feats,
            List<Tensor> current_vision_pos_embeds,
            List<(long, long)> feat_sizes,
            (Tensor,Tensor) point_inputs,
            Tensor mask_inputs,
            Dictionary<string, object> output_dict,
            int? num_frames,
            bool track_in_reverse = false,
            bool run_mem_encoder = true,
            Tensor prev_sam_mask_logits = null)
        {
            // 1. 调用内部跟踪步骤
            var (current_out, sam_outputs, _, _) = _track_step(
                frame_idx,
                is_init_cond_frame,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                point_inputs,
                mask_inputs,
                output_dict,
                num_frames,
                track_in_reverse,
                prev_sam_mask_logits);

            // 2. 解析SAM输出结果
            var (_, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits) = sam_outputs;

            // 3. 写入核心预测结果到输出字典
            current_out["pred_masks"] = low_res_masks;
            current_out["pred_masks_high_res"] = high_res_masks;
            current_out["obj_ptr"] = obj_ptr;

            // 4. 推理阶段额外写入对象分数logits（避免训练时的激活检查点未使用参数）
            if (!training)
            {
                current_out["object_score_logits"] = object_score_logits;
            }

            // 5. 编码内存特征
            _encode_memory_in_output(
                stack(current_vision_feats),
                feat_sizes,
                point_inputs,
                run_mem_encoder,
                high_res_masks,
                object_score_logits,
                current_out);

            return current_out;
        }

        /// <summary>
        /// </summary>
        //public Dictionary<string, object> track_step(
        //    int frame_idx,
        //    bool is_init_cond_frame,
        //    List<Tensor> current_vision_feats,
        //    List<Tensor> current_vision_pos_embeds,
        //    List<(long, long)> feat_sizes,
        //    object point_inputs,
        //    Tensor mask_inputs,
        //    Dictionary<string, object> output_dict,
        //    int? num_frames,
        //    bool track_in_reverse = false,  // tracking in reverse time order (for demo usage)
        //                                    // Whether to run the memory encoder on the predicted masks. Sometimes we might want
        //                                    // to skip the memory encoder with `run_mem_encoder=False`. For example,
        //                                    // in demo we might call `track_step` multiple times for each user click,
        //                                    // and only encode the memory when the user finalizes their clicks. And in ablation
        //                                    // settings like SAM training on static images, we don't need the memory encoder.
        //    bool run_mem_encoder = true,
        //    // The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        //    Tensor prev_sam_mask_logits = null)
        //{
        //    // 1. 调用内部跟踪步骤
        //    var (current_out, sam_outputs, _, _) = _track_step(
        //        frame_idx,
        //        is_init_cond_frame,
        //        current_vision_feats,
        //        current_vision_pos_embeds,
        //        feat_sizes,
        //        point_inputs,
        //        mask_inputs,
        //        output_dict,
        //        num_frames,
        //        track_in_reverse,
        //        prev_sam_mask_logits);

        //    // 2. 解析SAM输出结果（7元组解构）
        //    var (_, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits) = sam_outputs;

        //    // 3. 写入核心预测结果到输出字典
        //    current_out["pred_masks"] = low_res_masks;
        //    current_out["pred_masks_high_res"] = high_res_masks;
        //    current_out["obj_ptr"] = obj_ptr;

        //    // 4. 推理阶段额外写入对象分数logits（避免训练时的激活检查点未使用参数）
        //    if (!training)
        //    {
        //        // Only add this in inference (to avoid unused param in activation checkpointing;
        //        // it's mainly used in the demo to encode spatial memories w/ consolidated masks)
        //        current_out["object_score_logits"] = object_score_logits;
        //    }

        //    // 5. 编码内存特征（用于后续帧跟踪）
        //    _encode_memory_in_output(
        //        current_vision_feats,
        //        feat_sizes,
        //        point_inputs,
        //        run_mem_encoder,
        //        high_res_masks,
        //        object_score_logits,
        //        current_out);

        //    return current_out;
        //}

        /// <summary>
        /// Whether to use multimask output in the SAM head.
        /// </summary>
        private bool _use_multimask(bool is_init_cond_frame, (Tensor, Tensor) point_inputs)
        {
            // 计算点的数量
            int num_pts = 0;
            if (point_inputs.Item1 is not null)
            {
                var pointLabels = point_inputs.Item2;
                num_pts = (int)pointLabels.size(1);
            }

            // 判断是否输出多掩码
            bool multimask_output = (
                multimask_output_in_sam
                && (is_init_cond_frame || multimask_output_for_tracking)
                && (multimask_min_pt_num <= num_pts && num_pts <= multimask_max_pt_num)
            );

            return multimask_output;
        }

        /// <summary>
        /// Apply non-overlapping constraints to the object scores in pred_masks. Here we
        /// keep only the highest scoring object at each spatial location in pred_masks.
        /// </summary>
        private Tensor _apply_non_overlapping_constraints(Tensor pred_masks)
        {
            // 1. 获取批次大小，批次为1时直接返回（无需非重叠约束）
            long batch_size = pred_masks.size(0);
            if (batch_size == 1)
            {
                return pred_masks;
            }

            Device device = pred_masks.device;

            // 2. 计算每个空间位置得分最高的对象索引 [1, H, W]
            var max_obj_inds = argmax(pred_masks, dim: 0, keepdim: true);

            // 3. 生成每个对象切片的索引 [B, 1, 1, 1]
            var batch_obj_inds = arange(batch_size, device: device)
                .unsqueeze(1).unsqueeze(1).unsqueeze(1);

            // 4. 判断每个位置是否保留当前对象的得分
            var keep = max_obj_inds == batch_obj_inds;

            // 5. 抑制重叠区域的得分（设为-10以下，sigmoid(-10)=4.5398e-05，接近0）
            var clampedMasks = clamp(pred_masks, max: -10.0f);
            pred_masks = where(keep, pred_masks, clampedMasks);

            return pred_masks;
        }

        #endregion

        #region 10. 资源释放：TorchSharp非托管资源管理
        private bool _disposed = false;

        protected override void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // 释放托管子模块
                    image_encoder?.Dispose();
                    sam_mask_decoder?.Dispose();
                    memory_attention?.Dispose();
                    memory_encoder?.Dispose();
                    obj_ptr_proj?.Dispose();
                    obj_ptr_tpos_proj?.Dispose();

                    // 释放参数与张量
                    no_mem_embed?.Dispose();
                    no_mem_pos_enc?.Dispose();
                    no_obj_embed_spatial?.Dispose();
                    maskmem_tpos_enc?.Dispose();
                }

                //// 释放非托管资源
                //this.Parameters().ForEach(p => p?.Dispose());
                //this.Buffers().ForEach(b => b?.Dispose());

                _disposed = true;
                base.Dispose(disposing);
            }
        }

        ~Sam2Base()
        {
            Dispose(false);
        }
        #endregion
    }
}
