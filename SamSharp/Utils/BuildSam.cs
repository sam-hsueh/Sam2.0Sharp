using OpenCvSharp.Dnn;
using OpenCvSharp.ML;
using Sam2Sharp.Modeling;
using Sam2Sharp.Modeling.Backbones;
using Sam2Sharp.Modeling.PositionEncoding;
using Sam2Sharp.Tools;
using Sam2Sharp.Utils;
using System.Formats.Asn1;
using TorchSharp;
using TorchSharp.Modules;
using static Sam2Sharp.Modeling.Sam2.Transformer;
using static Sam2Sharp.Utils.Classes;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Sam2Sharp.Modeling.Sam2;
namespace Sam2Sharp.Utils
{
	internal class BuildSam2
	{
       
        internal static Sam2Base BuildSam2Model(string checkpoint, Device device, ScalarType dtype)
		{
			List<CommonTensor> commonTensors = PickleLoader.ReadTensorsInfoFromFile(checkpoint);
			return commonTensors.Count switch
			{
                //903 => build_sam_vit_t21l(checkpoint, device, dtype, commonTensors),//sam2.1l
                615 => build_sam_vit_t21b(checkpoint, device, dtype, commonTensors),//sam2.1b
                519 => build_sam_vit_t21s(checkpoint, device, dtype, commonTensors),//sam2.1s
                516 => build_sam_vit_t20s(checkpoint, device, dtype, commonTensors),//sam2s
                471 => build_sam_vit_t21t(checkpoint, device, dtype, commonTensors),//sam2.1
                468 => build_sam_vit_t20t(checkpoint, device, dtype, commonTensors),//sam2

                _ => throw new ArgumentException("Invalid SAM type specified.")
			};
		}
         static  int image_size = 1024;
        private static Sam2Base build_sam_vit_t20s(string checkpoint, torch.Device device, torch.ScalarType dtype, List<CommonTensor> commonTensors)
        {
            int prompt_embed_dim = 256;
            int vit_patch_size = 16;
            int image_embedding_size = image_size / vit_patch_size;

            // Model
            //  model:
            //  _target_: sam2.modeling.sam2_base.SAM2Base
            var image_encoder2 = new ImageEncoder(
       scalp: 1,
             trunk:
                   new Hiera(
                       embed_dim: 96,
         num_heads: 1,
         stages: [1, 2, 11, 2],
         global_att_blocks: [7, 10, 13],
         window_pos_embed_bkg_spatial_size: [7, 7]),
       neck:
           new FpnNeck(
                   position_encoding: new PositionEmbeddingSine(
           num_pos_feats: 256,
           temperature: 10000,
           normalize: true,
           scale: null,
           image_size: image_size
           ),
         d_model: 256,
         dtype: dtype,
         backbone_channel_list: [768, 384, 192, 96],
         fpn_top_down_levels: [2],  // output level 0 and 1 directly use the backbone features
         fpn_interp_model: InterpolationMode.Nearest));
            var memory_attention =
                new MemoryAttention(
              d_model: 256,
              pos_enc_at_input: true,
              layer:
                  new MemoryAttentionLayer(
                  activation: functional.relu,
                  dim_feedforward: 2048,
                dropout: 0.1f,
                pos_enc_at_attn: false,
                self_attention: new RoPEAttention(
                  rope_theta: 10000.0f,
                  feat_sizes: [64, 64],
                  embedding_dim: 256,
                  num_heads: 1,
                  downsample_rate: 1,
                  dropout: 0.1f),
                d_model: 256,
                pos_enc_at_cross_attn_keys: true,
                pos_enc_at_cross_attn_queries: false,
                cross_attention: new RoPEAttention(
                  rope_theta: 10000.0f,
                  feat_sizes: [64, 64],
                  rope_k_repeat: true,
                  embedding_dim: 256,
                  num_heads: 1,
                  downsample_rate: 1,
                  dropout: 0.1f,
                  kv_in_dim: 64)),
              num_layers: 4);
            var memory_encoder = new MemoryEncoder(
                  out_dim: 64,
                position_encoding:
                   new PositionEmbeddingSine(
                  num_pos_feats: 64,
                  normalize: true,
                  scale: null,
                  image_size: image_size,
                  temperature: 10000),
                mask_downsampler: new MaskDownSampler(
                  kernel_size: 3,
                  stride: 2,
                  padding: 1),
                fuser:
                  new Fuser(
                  layer: new CXBlock(
                    dim: 256,
                    kernel_size: 7,
                    padding: 3,
                    layer_scale_init_value: 1e-6f,
                    use_dwconv: true),  // depth-wise convs
                  num_layers: 2));
            //    sam2_prompt_encoder = PromptEncoder(
            //    embed_dim = self.sam_prompt_embed_dim,
            //    image_embedding_size = (
            //        self.sam_image_embedding_size,
            //        self.sam_image_embedding_size,
            //    ),
            //    input_image_size = (self.image_size, self.image_size),
            //    mask_in_chans = 16,
            //)
            //self.sam_mask_decoder = MaskDecoder(
            //    num_multimask_outputs = 3,
            //    transformer = TwoWayTransformer(
            //        depth = 2,
            //        embedding_dim = self.sam_prompt_embed_dim,
            //        mlp_dim = 2048,
            //        num_heads = 8,
            //    ),
            //    transformer_dim = self.sam_prompt_embed_dim,
            //    iou_head_depth = 3,
            //    iou_head_hidden_dim = 256,
            //    use_high_res_features = self.use_high_res_features_in_sam,
            //    iou_prediction_use_sigmoid = self.iou_prediction_use_sigmoid,
            //    pred_obj_scores = self.pred_obj_scores,
            //    pred_obj_scores_mlp = self.pred_obj_scores_mlp,
            //    use_multimask_token_for_obj_ptr = self.use_multimask_token_for_obj_ptr,
            //    **(self.sam_mask_decoder_extra_args or { }),
            //)

            //Sam2Sharp.Modeling.Sam.PromptEncoder promptEncoder = new Sam2Sharp.Modeling.Sam.PromptEncoder(
            //    embed_dim: prompt_embed_dim,
            //    image_embedding_size: (image_embedding_size, image_embedding_size),
            //    input_image_size: (image_size, image_size),
            //    mask_in_chans: 16).to(device, dtype);

            //Sam2Sharp.Modeling.Sam.MaskDecoder maskDecoder = new Sam2Sharp.Modeling.Sam.MaskDecoder(
            //    num_multimask_outputs: 3,
            //    transformer: new Sam2Sharp.Modeling.Sam2.Transformer.TwoWayTransformer(
            //        depth: 2,
            //        embedding_dim: prompt_embed_dim,
            //        mlp_dim: 2048,
            //        num_heads: 8),
            //    transformer_dim: prompt_embed_dim,
            //    iou_head_depth: 3,
            //    iou_head_hidden_dim: 256).to(device, dtype);

            Sam2Base sam = new Sam2Base(image_encoder2, memory_attention, memory_encoder, num_maskmem: 7,
  image_size: image_size,
  // apply scaled sigmoid on mask logits for memory encoder, and directly feed input mask as output mask
  // SAM decoder
  sigmoid_scale_for_mem_enc: 20.0f,
  sigmoid_bias_for_mem_enc: -10.0f,
  use_mask_input_as_output_without_sam: true,
  // Memory
  directly_add_no_mem_embed: true,
  // use high-resolution feature map in the SAM mask decoder
  use_high_res_features_in_sam: true,
  //True时，MaskDecoder会输出3个mask,两个卷积层的权重会被加载进去
  // output 3 masks on the first click on initial conditioning frames
  multimask_output_in_sam: false,
  // SAM heads
  iou_prediction_use_sigmoid: true,
  // cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
  use_obj_ptrs_in_encoder: true,
  add_tpos_enc_to_obj_ptrs: false,
  //proj_tpos_enc_in_obj_ptrs: true,
  //use_signed_tpos_enc_to_obj_ptrs: true,
  only_obj_ptrs_in_the_past_for_eval: true,
  // object occlusion prediction
  pred_obj_scores: true,
  pred_obj_scores_mlp: true,
  fixed_no_obj_ptr: true,
  // multimask tracking settings
  multimask_output_for_tracking: true,
  use_multimask_token_for_obj_ptr: true,
  multimask_min_pt_num: 0,
  multimask_max_pt_num: 1,
  use_mlp_for_obj_ptr_proj: true,
  // Compilation flag
  // HieraT does not currently support compilation, should always be set to False
  compile_image_encoder: false);       

            if (!string.IsNullOrEmpty(checkpoint))
            {
                Dictionary<string, Tensor> state_dict = PickleLoader.Load(checkpoint);
                (var error, var missing) = sam.load_state_dict(state_dict, strict: true);
                if (error.Count + missing.Count > 0)
                {
                    throw new ArgumentException("Error loading state dict");
                }
            }
            return sam.to(device, dtype);
        }
        private static Sam2Base build_sam_vit_t21l(string checkpoint, torch.Device device, torch.ScalarType dtype, List<CommonTensor> commonTensors)
        {
            int prompt_embed_dim = 256;
            int vit_patch_size = 16;
            int image_embedding_size = image_size / vit_patch_size;

            // Model
            //  model:
            //  _target_: sam2.modeling.sam2_base.SAM2Base
            var image_encoder2 = new ImageEncoder(
       scalp: 1,
             trunk:
                   new Hiera(
                       embed_dim: 144,
         num_heads: 2,
         stages: [2, 6, 36, 4],
         global_att_blocks: [23, 33, 43],
         window_pos_embed_bkg_spatial_size: [7, 7],
         window_spec: [8, 4, 16, 8]),
         neck:
           new FpnNeck(
                   position_encoding: new PositionEmbeddingSine(
           num_pos_feats: 256,
           temperature: 10000,
           normalize: true,
           scale: null,
           image_size: image_size
           ),
         d_model: 256,
         dtype: dtype,
         backbone_channel_list: [1152, 576, 288, 144],
         fpn_top_down_levels: [2],  // output level 0 and 1 directly use the backbone features
         fpn_interp_model: InterpolationMode.Nearest));
            var memory_attention =
                new MemoryAttention(
              d_model: 256,
              pos_enc_at_input: true,
              layer:
                  new MemoryAttentionLayer(
                  activation: functional.relu,
                  dim_feedforward: 2048,
                dropout: 0.1f,
                pos_enc_at_attn: false,
                self_attention: new RoPEAttention(
                  rope_theta: 10000.0f,
                  feat_sizes: [64, 64],
                  embedding_dim: 256,
                  num_heads: 1,
                  downsample_rate: 1,
                  dropout: 0.1f),
                d_model: 256,
                pos_enc_at_cross_attn_keys: true,
                pos_enc_at_cross_attn_queries: false,
                cross_attention: new RoPEAttention(
                  rope_theta: 10000.0f,
                  feat_sizes: [64, 64],
                  rope_k_repeat: true,
                  embedding_dim: 256,
                  num_heads: 1,
                  downsample_rate: 1,
                  dropout: 0.1f,
                  kv_in_dim: 64)),
              num_layers: 4);
            var memory_encoder = new MemoryEncoder(
                  out_dim: 64,
                position_encoding:
                   new PositionEmbeddingSine(
                  num_pos_feats: 64,
                  normalize: true,
                  scale: null,
                  image_size: image_size,
                  temperature: 10000),
                mask_downsampler: new MaskDownSampler(
                  kernel_size: 3,
                  stride: 2,
                  padding: 1),
                fuser:
                  new Fuser(
                  layer: new CXBlock(
                    dim: 256,
                    kernel_size: 7,
                    padding: 3,
                    layer_scale_init_value: 1e-6f,
                    use_dwconv: true),  // depth-wise convs
                  num_layers: 2));
            //    sam2_prompt_encoder = PromptEncoder(
            //    embed_dim = self.sam_prompt_embed_dim,
            //    image_embedding_size = (
            //        self.sam_image_embedding_size,
            //        self.sam_image_embedding_size,
            //    ),
            //    input_image_size = (self.image_size, self.image_size),
            //    mask_in_chans = 16,
            //)
            //self.sam_mask_decoder = MaskDecoder(
            //    num_multimask_outputs = 3,
            //    transformer = TwoWayTransformer(
            //        depth = 2,
            //        embedding_dim = self.sam_prompt_embed_dim,
            //        mlp_dim = 2048,
            //        num_heads = 8,
            //    ),
            //    transformer_dim = self.sam_prompt_embed_dim,
            //    iou_head_depth = 3,
            //    iou_head_hidden_dim = 256,
            //    use_high_res_features = self.use_high_res_features_in_sam,
            //    iou_prediction_use_sigmoid = self.iou_prediction_use_sigmoid,
            //    pred_obj_scores = self.pred_obj_scores,
            //    pred_obj_scores_mlp = self.pred_obj_scores_mlp,
            //    use_multimask_token_for_obj_ptr = self.use_multimask_token_for_obj_ptr,
            //    **(self.sam_mask_decoder_extra_args or { }),
            //)

            //Sam2Sharp.Modeling.Sam.PromptEncoder promptEncoder = new Sam2Sharp.Modeling.Sam.PromptEncoder(
            //    embed_dim: prompt_embed_dim,
            //    image_embedding_size: (image_embedding_size, image_embedding_size),
            //    input_image_size: (image_size, image_size),
            //    mask_in_chans: 16).to(device, dtype);

            //Sam2Sharp.Modeling.Sam.MaskDecoder maskDecoder = new Sam2Sharp.Modeling.Sam.MaskDecoder(
            //    num_multimask_outputs: 3,
            //    transformer: new Sam2Sharp.Modeling.Sam2.Transformer.TwoWayTransformer(
            //        depth: 2,
            //        embedding_dim: prompt_embed_dim,
            //        mlp_dim: 2048,
            //        num_heads: 8),
            //    transformer_dim: prompt_embed_dim,
            //    iou_head_depth: 3,
            //    iou_head_hidden_dim: 256).to(device, dtype);

            Sam2Base sam = new Sam2Base(image_encoder2, memory_attention, memory_encoder, num_maskmem: 7,
  image_size: image_size,
  // apply scaled sigmoid on mask logits for memory encoder, and directly feed input mask as output mask
  // SAM decoder
  sigmoid_scale_for_mem_enc: 20.0f,
  sigmoid_bias_for_mem_enc: -10.0f,
  use_mask_input_as_output_without_sam: true,
  // Memory
  directly_add_no_mem_embed: true,
  no_obj_embed_spatial: true,
  // use high-resolution feature map in the SAM mask decoder
  use_high_res_features_in_sam: true,
  // output 3 masks on the first click on initial conditioning frames
  multimask_output_in_sam: false,
  // SAM heads
  iou_prediction_use_sigmoid: true,
  // cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
  use_obj_ptrs_in_encoder: true,
  add_tpos_enc_to_obj_ptrs: true,
  proj_tpos_enc_in_obj_ptrs: true,
  use_signed_tpos_enc_to_obj_ptrs: true,
  only_obj_ptrs_in_the_past_for_eval: true,
  // object occlusion prediction
  pred_obj_scores: true,
  pred_obj_scores_mlp: true,
  fixed_no_obj_ptr: true,
  // multimask tracking settings
  multimask_output_for_tracking: true,
  use_multimask_token_for_obj_ptr: true,
  multimask_min_pt_num: 0,
  multimask_max_pt_num: 1,
  use_mlp_for_obj_ptr_proj: true,
  // Compilation flag
  // HieraT does not currently support compilation, should always be set to False
  compile_image_encoder: false);

            if (!string.IsNullOrEmpty(checkpoint))
            {
                Dictionary<string, Tensor> state_dict = PickleLoader.Load(checkpoint);
                (var error, var missing) = sam.load_state_dict(state_dict, strict: true);
                if (error.Count + missing.Count > 0)
                {
                    throw new ArgumentException("Error loading state dict");
                }
            }

            //Dictionary<string, Tensor> state_dict2 = PickleLoader.Load("D:\\MODELS\\SAM\\mobile_sam.pt");
            //(var error1, var missing1) = promptEncoder.load_state_dict(state_dict2, strict: false, prefix: "prompt_encoder.");
            ////if (error1.Count + missing1.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //(var error2, var missing2) = maskDecoder.load_state_dict(state_dict2, strict: false, prefix: "mask_decoder.");
            ////if (error2.Count + missing2.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //sam.sam2Sharp_prompt_encoder = promptEncoder;
            //sam.sam2Sharp_mask_decoder = maskDecoder;
            //var aa = torch.jit.load("D:\\MODELS\\SAM\\Prompt_guided_Mask_Decoder.pt");
            //Dictionary<string, Tensor> state_dict2 = PickleLoader.Load("D:\\MODELS\\SAM\\Prompt_guided_Mask_Decoder.pt");
            //(var error1, var missing1) = promptEncoder.load_state_dict(state_dict2, strict: false, prefix: "prompt_encoder.");
            ////if (error1.Count + missing1.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //(var error2, var missing2) = maskDecoder.load_state_dict(state_dict2, strict: false, prefix: "mask_decoder.");
            ////if (error2.Count + missing2.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            return sam.to(device, dtype);
        }
        private static Sam2Base build_sam_vit_t21b(string checkpoint, torch.Device device, torch.ScalarType dtype, List<CommonTensor> commonTensors)
        {
            int prompt_embed_dim = 256;
            int vit_patch_size = 16;
            int image_embedding_size = image_size / vit_patch_size;

            // Model
            //  model:
            //  _target_: sam2.modeling.sam2_base.SAM2Base
            var image_encoder2 = new ImageEncoder(
       scalp: 1,
             trunk:
                   new Hiera(
                       embed_dim: 112,
         num_heads: 2
         //stages: [2, 6, 36, 4],
         //global_att_blocks: [23, 33, 43],
         /*window_pos_embed_bkg_spatial_size: [7, 7]*/),

         neck:
           new FpnNeck(
                   position_encoding: new PositionEmbeddingSine(
           num_pos_feats: 256,
           temperature: 10000,
           normalize: true,
           scale: null,
           image_size: image_size
           ),
         d_model: 256,
         dtype: dtype,
         backbone_channel_list: [896, 448, 224, 112],
         fpn_top_down_levels: [2],  // output level 0 and 1 directly use the backbone features
         fpn_interp_model: InterpolationMode.Nearest));
            var memory_attention =
                new MemoryAttention(
              d_model: 256,
              pos_enc_at_input: true,
              layer:
                  new MemoryAttentionLayer(
                  activation: functional.relu,
                  dim_feedforward: 2048,
                dropout: 0.1f,
                pos_enc_at_attn: false,
                self_attention: new RoPEAttention(
                  rope_theta: 10000.0f,
                  feat_sizes: [64, 64],
                  embedding_dim: 256,
                  num_heads: 1,
                  downsample_rate: 1,
                  dropout: 0.1f),
                d_model: 256,
                pos_enc_at_cross_attn_keys: true,
                pos_enc_at_cross_attn_queries: false,
                cross_attention: new RoPEAttention(
                  rope_theta: 10000.0f,
                  feat_sizes: [64, 64],
                  rope_k_repeat: true,
                  embedding_dim: 256,
                  num_heads: 1,
                  downsample_rate: 1,
                  dropout: 0.1f,
                  kv_in_dim: 64)),
              num_layers: 4);
            var memory_encoder = new MemoryEncoder(
                  out_dim: 64,
                position_encoding:
                   new PositionEmbeddingSine(
                  num_pos_feats: 64,
                  normalize: true,
                  scale: null,
                  image_size: image_size,
                  temperature: 10000),
                mask_downsampler: new MaskDownSampler(
                  kernel_size: 3,
                  stride: 2,
                  padding: 1),
                fuser:
                  new Fuser(
                  layer: new CXBlock(
                    dim: 256,
                    kernel_size: 7,
                    padding: 3,
                    layer_scale_init_value: 1e-6f,
                    use_dwconv: true),  // depth-wise convs
                  num_layers: 2));
            //    sam2_prompt_encoder = PromptEncoder(
            //    embed_dim = self.sam_prompt_embed_dim,
            //    image_embedding_size = (
            //        self.sam_image_embedding_size,
            //        self.sam_image_embedding_size,
            //    ),
            //    input_image_size = (self.image_size, self.image_size),
            //    mask_in_chans = 16,
            //)
            //self.sam_mask_decoder = MaskDecoder(
            //    num_multimask_outputs = 3,
            //    transformer = TwoWayTransformer(
            //        depth = 2,
            //        embedding_dim = self.sam_prompt_embed_dim,
            //        mlp_dim = 2048,
            //        num_heads = 8,
            //    ),
            //    transformer_dim = self.sam_prompt_embed_dim,
            //    iou_head_depth = 3,
            //    iou_head_hidden_dim = 256,
            //    use_high_res_features = self.use_high_res_features_in_sam,
            //    iou_prediction_use_sigmoid = self.iou_prediction_use_sigmoid,
            //    pred_obj_scores = self.pred_obj_scores,
            //    pred_obj_scores_mlp = self.pred_obj_scores_mlp,
            //    use_multimask_token_for_obj_ptr = self.use_multimask_token_for_obj_ptr,
            //    **(self.sam_mask_decoder_extra_args or { }),
            //)

            //Sam2Sharp.Modeling.Sam.PromptEncoder promptEncoder = new Sam2Sharp.Modeling.Sam.PromptEncoder(
            //    embed_dim: prompt_embed_dim,
            //    image_embedding_size: (image_embedding_size, image_embedding_size),
            //    input_image_size: (image_size, image_size),
            //    mask_in_chans: 16).to(device, dtype);

            //Sam2Sharp.Modeling.Sam.MaskDecoder maskDecoder = new Sam2Sharp.Modeling.Sam.MaskDecoder(
            //    num_multimask_outputs: 3,
            //    transformer: new Sam2Sharp.Modeling.Sam2.Transformer.TwoWayTransformer(
            //        depth: 2,
            //        embedding_dim: prompt_embed_dim,
            //        mlp_dim: 2048,
            //        num_heads: 8),
            //    transformer_dim: prompt_embed_dim,
            //    iou_head_depth: 3,
            //    iou_head_hidden_dim: 256).to(device, dtype);

            Sam2Base sam = new Sam2Base(image_encoder2, memory_attention, memory_encoder, num_maskmem: 7,
  image_size: image_size,
  // apply scaled sigmoid on mask logits for memory encoder, and directly feed input mask as output mask
  // SAM decoder
  sigmoid_scale_for_mem_enc: 20.0f,
  sigmoid_bias_for_mem_enc: -10.0f,
  use_mask_input_as_output_without_sam: true,
  // Memory
  directly_add_no_mem_embed: true,
  no_obj_embed_spatial: true,
  // use high-resolution feature map in the SAM mask decoder
  use_high_res_features_in_sam: true,
  // output 3 masks on the first click on initial conditioning frames
  multimask_output_in_sam: false,
  // SAM heads
  iou_prediction_use_sigmoid: true,
  // cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
  use_obj_ptrs_in_encoder: true,
  add_tpos_enc_to_obj_ptrs: true,
  proj_tpos_enc_in_obj_ptrs: true,
  use_signed_tpos_enc_to_obj_ptrs: true,
  only_obj_ptrs_in_the_past_for_eval: true,
  // object occlusion prediction
  pred_obj_scores: true,
  pred_obj_scores_mlp: true,
  fixed_no_obj_ptr: true,
  // multimask tracking settings
  multimask_output_for_tracking: true,
  use_multimask_token_for_obj_ptr: true,
  multimask_min_pt_num: 0,
  multimask_max_pt_num: 1,
  use_mlp_for_obj_ptr_proj: true,
  // Compilation flag
  // HieraT does not currently support compilation, should always be set to False
  compile_image_encoder: false);

            if (!string.IsNullOrEmpty(checkpoint))
            {
                Dictionary<string, Tensor> state_dict = PickleLoader.Load(checkpoint);
                (var error, var missing) = sam.load_state_dict(state_dict, strict: true);
                if (error.Count + missing.Count > 0)
                {
                    throw new ArgumentException("Error loading state dict");
                }
            }

            //Dictionary<string, Tensor> state_dict2 = PickleLoader.Load("D:\\MODELS\\SAM\\mobile_sam.pt");
            //(var error1, var missing1) = promptEncoder.load_state_dict(state_dict2, strict: false, prefix: "prompt_encoder.");
            ////if (error1.Count + missing1.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //(var error2, var missing2) = maskDecoder.load_state_dict(state_dict2, strict: false, prefix: "mask_decoder.");
            ////if (error2.Count + missing2.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //sam.sam2Sharp_prompt_encoder = promptEncoder;
            //sam.sam2Sharp_mask_decoder = maskDecoder;
            //var aa = torch.jit.load("D:\\MODELS\\SAM\\Prompt_guided_Mask_Decoder.pt");
            //Dictionary<string, Tensor> state_dict2 = PickleLoader.Load("D:\\MODELS\\SAM\\Prompt_guided_Mask_Decoder.pt");
            //(var error1, var missing1) = promptEncoder.load_state_dict(state_dict2, strict: false, prefix: "prompt_encoder.");
            ////if (error1.Count + missing1.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //(var error2, var missing2) = maskDecoder.load_state_dict(state_dict2, strict: false, prefix: "mask_decoder.");
            ////if (error2.Count + missing2.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            return sam.to(device, dtype);
        }

        private static Sam2Base build_sam_vit_t21s(string checkpoint, torch.Device device, torch.ScalarType dtype, List<CommonTensor> commonTensors)
        {
            int prompt_embed_dim = 256;
            int vit_patch_size = 16;
            int image_embedding_size = image_size / vit_patch_size;

            // Model
            //  model:
            //  _target_: sam2.modeling.sam2_base.SAM2Base
            var image_encoder2 = new ImageEncoder(
       scalp: 1,
             trunk:
                   new Hiera(
                       embed_dim: 96,
         num_heads: 1,
         stages: [1, 2, 11, 2],
         global_att_blocks: [7, 10, 13],
         window_pos_embed_bkg_spatial_size: [7, 7]),

       neck:
           new FpnNeck(
                   position_encoding: new PositionEmbeddingSine(
           num_pos_feats: 256,
           temperature: 10000,
           normalize: true,
           scale: null,
           image_size: image_size
           ),
         d_model: 256,
         dtype: dtype,
         backbone_channel_list: [768, 384, 192, 96],
         fpn_top_down_levels: [2],  // output level 0 and 1 directly use the backbone features
         fpn_interp_model: InterpolationMode.Nearest));
            var memory_attention =
                new MemoryAttention(
              d_model: 256,
              pos_enc_at_input: true,
              layer:
                  new MemoryAttentionLayer(
                  activation: functional.relu,
                  dim_feedforward: 2048,
                dropout: 0.1f,
                pos_enc_at_attn: false,
                self_attention: new RoPEAttention(
                  rope_theta: 10000.0f,
                  feat_sizes: [64, 64],
                  embedding_dim: 256,
                  num_heads: 1,
                  downsample_rate: 1,
                  dropout: 0.1f),
                d_model: 256,
                pos_enc_at_cross_attn_keys: true,
                pos_enc_at_cross_attn_queries: false,
                cross_attention: new RoPEAttention(
                  rope_theta: 10000.0f,
                  feat_sizes: [64, 64],
                  rope_k_repeat: true,
                  embedding_dim: 256,
                  num_heads: 1,
                  downsample_rate: 1,
                  dropout: 0.1f,
                  kv_in_dim: 64)),
              num_layers: 4);
            var memory_encoder = new MemoryEncoder(
                  out_dim: 64,
                position_encoding:
                   new PositionEmbeddingSine(
                  num_pos_feats: 64,
                  normalize: true,
                  scale: null,
                  image_size: image_size,
                  temperature: 10000),
                mask_downsampler: new MaskDownSampler(
                  kernel_size: 3,
                  stride: 2,
                  padding: 1),
                fuser:
                  new Fuser(
                  layer: new CXBlock(
                    dim: 256,
                    kernel_size: 7,
                    padding: 3,
                    layer_scale_init_value: 1e-6f,
                    use_dwconv: true),  // depth-wise convs
                  num_layers: 2));
            //    sam2_prompt_encoder = PromptEncoder(
            //    embed_dim = self.sam_prompt_embed_dim,
            //    image_embedding_size = (
            //        self.sam_image_embedding_size,
            //        self.sam_image_embedding_size,
            //    ),
            //    input_image_size = (self.image_size, self.image_size),
            //    mask_in_chans = 16,
            //)
            //self.sam_mask_decoder = MaskDecoder(
            //    num_multimask_outputs = 3,
            //    transformer = TwoWayTransformer(
            //        depth = 2,
            //        embedding_dim = self.sam_prompt_embed_dim,
            //        mlp_dim = 2048,
            //        num_heads = 8,
            //    ),
            //    transformer_dim = self.sam_prompt_embed_dim,
            //    iou_head_depth = 3,
            //    iou_head_hidden_dim = 256,
            //    use_high_res_features = self.use_high_res_features_in_sam,
            //    iou_prediction_use_sigmoid = self.iou_prediction_use_sigmoid,
            //    pred_obj_scores = self.pred_obj_scores,
            //    pred_obj_scores_mlp = self.pred_obj_scores_mlp,
            //    use_multimask_token_for_obj_ptr = self.use_multimask_token_for_obj_ptr,
            //    **(self.sam_mask_decoder_extra_args or { }),
            //)

            //Sam2Sharp.Modeling.Sam.PromptEncoder promptEncoder = new Sam2Sharp.Modeling.Sam.PromptEncoder(
            //    embed_dim: prompt_embed_dim,
            //    image_embedding_size: (image_embedding_size, image_embedding_size),
            //    input_image_size: (image_size, image_size),
            //    mask_in_chans: 16).to(device, dtype);

            //Sam2Sharp.Modeling.Sam.MaskDecoder maskDecoder = new Sam2Sharp.Modeling.Sam.MaskDecoder(
            //    num_multimask_outputs: 3,
            //    transformer: new Sam2Sharp.Modeling.Sam2.Transformer.TwoWayTransformer(
            //        depth: 2,
            //        embedding_dim: prompt_embed_dim,
            //        mlp_dim: 2048,
            //        num_heads: 8),
            //    transformer_dim: prompt_embed_dim,
            //    iou_head_depth: 3,
            //    iou_head_hidden_dim: 256).to(device, dtype);

            Sam2Base sam = new Sam2Base(image_encoder2, memory_attention, memory_encoder, num_maskmem: 7,
  image_size: image_size,
  // apply scaled sigmoid on mask logits for memory encoder, and directly feed input mask as output mask
  // SAM decoder
  sigmoid_scale_for_mem_enc: 20.0f,
  sigmoid_bias_for_mem_enc: -10.0f,
  use_mask_input_as_output_without_sam: true,
  // Memory
  directly_add_no_mem_embed: true,
  no_obj_embed_spatial: true,
  // use high-resolution feature map in the SAM mask decoder
  use_high_res_features_in_sam: true,
  // output 3 masks on the first click on initial conditioning frames
  multimask_output_in_sam: false,
  // SAM heads
  iou_prediction_use_sigmoid: true,
  // cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
  use_obj_ptrs_in_encoder: true,
  add_tpos_enc_to_obj_ptrs: true,
  proj_tpos_enc_in_obj_ptrs: true,
  use_signed_tpos_enc_to_obj_ptrs: true,
  only_obj_ptrs_in_the_past_for_eval: true,
  // object occlusion prediction
  pred_obj_scores: true,
  pred_obj_scores_mlp: true,
  fixed_no_obj_ptr: true,
  // multimask tracking settings
  multimask_output_for_tracking: true,
  use_multimask_token_for_obj_ptr: true,
  multimask_min_pt_num: 0,
  multimask_max_pt_num: 1,
  use_mlp_for_obj_ptr_proj: true,
  // Compilation flag
  // HieraT does not currently support compilation, should always be set to False
  compile_image_encoder: false);
       
            if (!string.IsNullOrEmpty(checkpoint))
            {
                Dictionary<string, Tensor> state_dict = PickleLoader.Load(checkpoint);
                (var error, var missing) = sam.load_state_dict(state_dict, strict: true);
                if (error.Count + missing.Count > 0)
                {
                    throw new ArgumentException("Error loading state dict");
                }
            }

            //Dictionary<string, Tensor> state_dict2 = PickleLoader.Load("D:\\MODELS\\SAM\\mobile_sam.pt");
            //(var error1, var missing1) = promptEncoder.load_state_dict(state_dict2, strict: false, prefix: "prompt_encoder.");
            ////if (error1.Count + missing1.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //(var error2, var missing2) = maskDecoder.load_state_dict(state_dict2, strict: false, prefix: "mask_decoder.");
            ////if (error2.Count + missing2.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //sam.sam2Sharp_prompt_encoder = promptEncoder;
            //sam.sam2Sharp_mask_decoder = maskDecoder;
            //var aa = torch.jit.load("D:\\MODELS\\SAM\\Prompt_guided_Mask_Decoder.pt");
            //Dictionary<string, Tensor> state_dict2 = PickleLoader.Load("D:\\MODELS\\SAM\\Prompt_guided_Mask_Decoder.pt");
            //(var error1, var missing1) = promptEncoder.load_state_dict(state_dict2, strict: false, prefix: "prompt_encoder.");
            ////if (error1.Count + missing1.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //(var error2, var missing2) = maskDecoder.load_state_dict(state_dict2, strict: false, prefix: "mask_decoder.");
            ////if (error2.Count + missing2.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            return sam.to(device, dtype);
        }

        private static Sam2Base build_sam_vit_t21t(string checkpoint, torch.Device device, torch.ScalarType dtype, List<CommonTensor> commonTensors)
        {
           // int prompt_embed_dim = 256;
            int vit_patch_size = 16;
            int image_embedding_size = image_size / vit_patch_size;

            var image_encoder2 = new ImageEncoder(
       scalp: 1,
             trunk:
                   new Hiera(
                       embed_dim: 96,
         num_heads: 1,
         stages: [1, 2, 7, 2],
         global_att_blocks: [5, 7, 9],
         window_pos_embed_bkg_spatial_size: [7, 7]),

       neck:
           new FpnNeck(
                   position_encoding: new PositionEmbeddingSine(
           num_pos_feats: 256,
           temperature: 10000,
           normalize: true,
           scale: null,
           image_size:image_size
           ),
         d_model: 256,
         dtype: dtype,
         backbone_channel_list: [768, 384, 192, 96],
         fpn_top_down_levels: [2,3],  // output level 0 and 1 & 3 directly use the backbone features
         fpn_interp_model: InterpolationMode.Nearest));

     var memory_attention =
         new MemoryAttention(
       d_model: 256,
       pos_enc_at_input: true,
       layer:
           new MemoryAttentionLayer(
           activation: functional.relu,
           dim_feedforward: 2048,
         dropout: 0.1f,
         pos_enc_at_attn: false,
         self_attention: new RoPEAttention(
           rope_theta: 10000.0f,
           feat_sizes: [64, 64],
           embedding_dim: 256,
           num_heads: 1,
           downsample_rate: 1,
           dropout: 0.1f),
         d_model: 256,
         pos_enc_at_cross_attn_keys: true,
         pos_enc_at_cross_attn_queries: false,
         cross_attention: new RoPEAttention(
           rope_theta: 10000.0f,
           feat_sizes: [64, 64],
           rope_k_repeat: true,
           embedding_dim: 256,
           num_heads: 1,
           downsample_rate: 1,
           dropout: 0.1f,
           kv_in_dim: 64)),
       num_layers: 4);

            var memory_encoder = new MemoryEncoder(
                  out_dim: 64,
                position_encoding:
                   new PositionEmbeddingSine(
                  num_pos_feats: 64,
                  normalize: true,
                  scale: null,
                  image_size:image_size,
                  temperature: 10000),
                mask_downsampler: new MaskDownSampler(
                  kernel_size: 3,
                  stride: 2,
                  padding: 1),
                fuser:
                  new Fuser(
                  layer: new CXBlock(
                    dim: 256,
                    kernel_size: 7,
                    padding: 3,
                    layer_scale_init_value: 1e-6f,
                    use_dwconv: true),  // depth-wise convs
                  num_layers: 2));

            //    sam2_prompt_encoder = PromptEncoder(
            //    embed_dim = self.sam_prompt_embed_dim,
            //    image_embedding_size = (
            //        self.sam_image_embedding_size,
            //        self.sam_image_embedding_size,
            //    ),
            //    input_image_size = (self.image_size, self.image_size),
            //    mask_in_chans = 16,
            //)
            //self.sam_mask_decoder = MaskDecoder(
            //    num_multimask_outputs = 3,
            //    transformer = TwoWayTransformer(
            //        depth = 2,
            //        embedding_dim = self.sam_prompt_embed_dim,
            //        mlp_dim = 2048,
            //        num_heads = 8,
            //    ),
            //    transformer_dim = self.sam_prompt_embed_dim,
            //    iou_head_depth = 3,
            //    iou_head_hidden_dim = 256,
            //    use_high_res_features = self.use_high_res_features_in_sam,
            //    iou_prediction_use_sigmoid = self.iou_prediction_use_sigmoid,
            //    pred_obj_scores = self.pred_obj_scores,
            //    pred_obj_scores_mlp = self.pred_obj_scores_mlp,
            //    use_multimask_token_for_obj_ptr = self.use_multimask_token_for_obj_ptr,
            //    **(self.sam_mask_decoder_extra_args or { }),
            //)

            //Sam2Sharp.Modeling.Sam.PromptEncoder promptEncoder = new Sam2Sharp.Modeling.Sam.PromptEncoder(
            //    embed_dim: prompt_embed_dim,
            //    image_embedding_size: (image_embedding_size, image_embedding_size),
            //    input_image_size: (image_size, image_size),
            //    mask_in_chans: 16).to(device, dtype);

            //Sam2Sharp.Modeling.Sam.MaskDecoder maskDecoder = new Sam2Sharp.Modeling.Sam.MaskDecoder(
            //    num_multimask_outputs: 3,
            //    transformer: new Sam2Sharp.Modeling.Sam2.Transformer.TwoWayTransformer(
            //        depth: 2,
            //        embedding_dim: prompt_embed_dim,
            //        mlp_dim: 2048,
            //        num_heads: 8),
            //    transformer_dim: prompt_embed_dim,
            //    iou_head_depth: 3,
            //    iou_head_hidden_dim: 256).to(device, dtype);

            Sam2Base sam = new Sam2Base(image_encoder2, memory_attention, memory_encoder,  num_maskmem: 7,
  image_size: image_size,
  // apply scaled sigmoid on mask logits for memory encoder, and directly feed input mask as output mask
  // SAM decoder
  sigmoid_scale_for_mem_enc: 20.0f,
  sigmoid_bias_for_mem_enc: -10.0f,
  use_mask_input_as_output_without_sam: true,
  // Memory
  directly_add_no_mem_embed: true,
  no_obj_embed_spatial: true,
  // use high-resolution feature map in the SAM mask decoder
  use_high_res_features_in_sam: true,
  // output 3 masks on the first click on initial conditioning frames
  multimask_output_in_sam: false,
  // SAM heads
  iou_prediction_use_sigmoid: true,
  // cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
  use_obj_ptrs_in_encoder: true,
  add_tpos_enc_to_obj_ptrs: true,
  proj_tpos_enc_in_obj_ptrs: true,
  use_signed_tpos_enc_to_obj_ptrs: true,
  only_obj_ptrs_in_the_past_for_eval: true,
  // object occlusion prediction
  pred_obj_scores: true,
  pred_obj_scores_mlp: true,
  fixed_no_obj_ptr: true,
  // multimask tracking settings
  multimask_output_for_tracking: true,
  use_multimask_token_for_obj_ptr: true,
  multimask_min_pt_num: 0,
  multimask_max_pt_num: 1,
  use_mlp_for_obj_ptr_proj: true,
  // Compilation flag
  // HieraT does not currently support compilation, should always be set to False
  compile_image_encoder: false);

            if (!string.IsNullOrEmpty(checkpoint))
            {
                Dictionary<string, Tensor> state_dict = PickleLoader.Load(checkpoint);
                (var error, var missing) = sam.load_state_dict(state_dict, strict: true);
                if (error.Count + missing.Count > 0)
                {
                    throw new ArgumentException("Error loading state dict");
                }
            }

            //Dictionary<string, Tensor> state_dict2 = PickleLoader.Load("D:\\MODELS\\SAM\\mobile_sam.pt");
            //(var error1, var missing1) = promptEncoder.load_state_dict(state_dict2, strict: false, prefix: "prompt_encoder.");
            ////if (error1.Count + missing1.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //(var error2, var missing2) = maskDecoder.load_state_dict(state_dict2, strict: false, prefix: "mask_decoder.");
            ////if (error2.Count + missing2.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //sam.sam2Sharp_prompt_encoder = promptEncoder;
            //sam.sam2Sharp_mask_decoder = maskDecoder;
            //var aa = torch.jit.load("D:\\MODELS\\SAM\\Prompt_guided_Mask_Decoder.pt");
            //Dictionary<string, Tensor> state_dict2 = PickleLoader.Load("D:\\MODELS\\SAM\\Prompt_guided_Mask_Decoder.pt");
            //(var error1, var missing1) = promptEncoder.load_state_dict(state_dict2, strict: false, prefix: "prompt_encoder.");
            ////if (error1.Count + missing1.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //(var error2, var missing2) = maskDecoder.load_state_dict(state_dict2, strict: false, prefix: "mask_decoder.");
            ////if (error2.Count + missing2.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            return sam.to(device, dtype);
        }

        private static Sam2Base build_sam_vit_t20t(string checkpoint, torch.Device device, torch.ScalarType dtype, List<CommonTensor> commonTensors)
		{
			int prompt_embed_dim = 256;
			int vit_patch_size = 16;
			int image_embedding_size = image_size / vit_patch_size;

            var image_encoder2 = new ImageEncoder(
                trunk : new Hiera(
				embed_dim: 96,
				num_heads: 1,
				stages: new[] { 1, 2, 7, 2 },
				global_att_blocks: new[] { 5, 7, 9 },
				window_pos_embed_bkg_spatial_size: [7, 7]),

                neck: new FpnNeck(
				position_encoding: new PositionEmbeddingSine(
                        num_pos_feats: 256,
            normalize: true,
            scale: 1,
            image_size:image_size,
            temperature: 10000
                ),
			d_model: 256,
            dtype: dtype,
            backbone_channel_list: [768, 384, 192, 96],
			fpn_top_down_levels: [2],  // output level 0 and 1 directly use the backbone features
			fpn_interp_model: InterpolationMode.Nearest),
                scalp: 1);

            var memory_attention = new MemoryAttention(
            d_model: 256,
            pos_enc_at_input: true,
            layer: new MemoryAttentionLayer(
                activation: functional.relu,
                cross_attention: new RoPEAttention(
            embedding_dim: 256,
            num_heads: 1,
            downsample_rate: 1,
            dropout: 0.1f,
             kv_in_dim: 64,
             rope_theta: 10000.0f,
             rope_k_repeat: true,
             feat_sizes: [64, 64]),
               d_model: 256,
               dim_feedforward: 2048,
              dropout: 0.1f,
              pos_enc_at_attn: false,
              pos_enc_at_cross_attn_keys: true,
              pos_enc_at_cross_attn_queries: false,
              self_attention: new RoPEAttention(
               rope_theta: 10000.0f,
               feat_sizes: [64, 64],
               embedding_dim: 256,
               num_heads: 1,
               downsample_rate: 1,
               dropout: 0.1f)
                ),
            num_layers: 4);

            var memory_encoder = new MemoryEncoder(
                  out_dim: 64,
                  mask_downsampler: new MaskDownSampler(
                  kernel_size: 3,
                  stride: 2,
                  padding: 1),
                  position_encoding: new PositionEmbeddingSine(
                  num_pos_feats: 64,
                  normalize: true,
                  scale: null,
                  image_size:image_size,
                  temperature: 10000),
                  fuser: new Fuser(
                    layer: new CXBlock(
                     dim: 256,
                     kernel_size: 7,
                     padding: 3,
                     layer_scale_init_value: 1e-6f,
                     use_dwconv: true),  // depth-wise convs
                   num_layers: 2));

            //var prompt_encoder = new Sam2Sharp.Modeling.Sam.PromptEncoder(
            //   embed_dim: prompt_embed_dim,
            //    image_embedding_size: (image_embedding_size, image_embedding_size),
            //    input_image_size: (image_size, image_size),
            //    mask_in_chans: 16).to(device, dtype);
            //var mask_decoder = new Sam2Sharp.Modeling.Sam.MaskDecoder(num_multimask_outputs: 3,
            //    transformer: new Sam2Sharp.Modeling.Transformer.TwoWayTransformer(
            //        depth: 2,
            //        embedding_dim: prompt_embed_dim,
            //        mlp_dim: 2048,
            //        num_heads: 8),
            //    transformer_dim: prompt_embed_dim,
            //    iou_head_depth: 3,
            //    iou_head_hidden_dim: 256).to(device, dtype);

            //        Sam2Sharp.Modeling.Sam.PromptEncoder prompt_encoder = new Sam2Sharp.Modeling.Sam.PromptEncoder(
            //embed_dim: prompt_embed_dim,
            //image_embedding_size: (image_embedding_size, image_embedding_size),
            //input_image_size: (image_size, image_size),
            //mask_in_chans: 16).to(device, dtype);

            //        Sam2Sharp.Modeling.Sam.MaskDecoder mask_eecoder = new Sam2Sharp.Modeling.Sam.MaskDecoder(
            //            num_multimask_outputs: 3,
            //            transformer: new Sam2Sharp.Modeling.Transformer.TwoWayTransformer(
            //                depth: 2,
            //                embedding_dim: prompt_embed_dim,
            //                mlp_dim: 2048,
            //                num_heads: 8),
            //            transformer_dim: prompt_embed_dim,
            //            iou_head_depth: 3,
            //            iou_head_hidden_dim: 256).to(device, dtype);

//            Sam2Sharp.Modeling.Sam.PromptEncoder promptEncoder = new Sam2Sharp.Modeling.Sam.PromptEncoder(
//embed_dim: prompt_embed_dim,
//image_embedding_size: (image_embedding_size, image_embedding_size),
//input_image_size: (image_size, image_size),
//mask_in_chans: 16).to(device, dtype);

//            Sam2Sharp.Modeling.Sam.MaskDecoder maskDecoder = new Sam2Sharp.Modeling.Sam.MaskDecoder(
//                num_multimask_outputs: 3,
//                transformer: new TwoWayTransformer(
//                    depth: 2,
//                    embedding_dim: prompt_embed_dim,
//                    mlp_dim: 2048,
//                    num_heads: 8),
//                transformer_dim: prompt_embed_dim,
//                iou_head_depth: 3,
//                iou_head_hidden_dim: 256).to(device, dtype);

            Sam2Base sam = new Sam2Base(image_encoder2, memory_attention, memory_encoder,
            //mask_downsample: torch.nn.Conv2d(16, 16, 1).to(device, dtype), // 示例：请替换为实际的 mask_downsample 实例
            //no_obj_ptr: torch.nn.Parameter(torch.zeros(new long[] { 1, 256 }, dtype: dtype, device: device)), // 示例：请替换为实际 no_obj_ptr
            //no_obj_embed_spatial: torch.nn.Parameter(torch.zeros(new long[] { 1, 64 }, dtype: dtype, device: device)), // 示例：请替换为实际 no_obj_embed_spatial
            num_maskmem: 7,
              image_size: image_size,
            // apply scaled sigmoid on mask logits for memory encoder, and directly feed input mask as output mask
            // SAM decoder
            sigmoid_scale_for_mem_enc: 20.0f,
            sigmoid_bias_for_mem_enc: -10.0f,
            use_mask_input_as_output_without_sam: true,
            // Memory
            directly_add_no_mem_embed: true,
            // use high-resolution feature map in the SAM mask decoder
            use_high_res_features_in_sam: true,
            // output 3 masks on the first click on initial conditioning frames
            multimask_output_in_sam: false,
            // SAM heads
            iou_prediction_use_sigmoid: true,
            // cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
            use_obj_ptrs_in_encoder: true,
            no_obj_embed_spatial: false,

            add_tpos_enc_to_obj_ptrs: false,
            only_obj_ptrs_in_the_past_for_eval: true,
            // object occlusion prediction
            pred_obj_scores: true,
            pred_obj_scores_mlp: true,
            fixed_no_obj_ptr: true,
            // multimask tracking settings
            multimask_output_for_tracking: true,
            use_multimask_token_for_obj_ptr: true,
            multimask_min_pt_num: 0,
            multimask_max_pt_num: 1,
            use_mlp_for_obj_ptr_proj: true,
            // Compilation flag
            // HieraT does not currently support compilation, should always be set to False
            compile_image_encoder: false).to(device,dtype);
            if (!string.IsNullOrEmpty(checkpoint))
            {
                Dictionary<string, Tensor> state_dict = PickleLoader.Load(checkpoint);
                (var error, var missing) = sam.load_state_dict(state_dict, strict: true);
                if (error.Count + missing.Count > 0)
                {
                    throw new ArgumentException("Error loading state dict");
                }
            }

            //Dictionary<string, Tensor> state_dict2 = PickleLoader.Load("D:\\MODELS\\SAM\\mobile_sam.pt");
            //(var error1, var missing1) = promptEncoder.load_state_dict(state_dict2, strict: false, prefix: "prompt_encoder.");
            ////if (error1.Count + missing1.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //(var error2, var missing2) = maskDecoder.load_state_dict(state_dict2, strict: false, prefix: "mask_decoder.");
            ////if (error2.Count + missing2.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //sam.sam2Sharp_prompt_encoder = promptEncoder;
            //sam.sam2Sharp_mask_decoder = maskDecoder;
            //Dictionary<string, Tensor> state_dict2 = PickleLoader.Load("D:\\MODELS\\SAM\\Prompt_guided_Mask_Decoder.pt");
            //(var error1, var missing1) = promptEncoder.load_state_dict(state_dict2, strict: false, prefix: "prompt_encoder.");
            ////if (error1.Count + missing1.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}
            //(var error2, var missing2) = maskDecoder.load_state_dict(state_dict2, strict: false, prefix: "mask_decoder.");
            ////if (error2.Count + missing2.Count > 0)
            ////{
            ////    throw new ArgumentException("Error loading state dict");
            ////}

            return sam.to(device, dtype);
		}
	}
}
