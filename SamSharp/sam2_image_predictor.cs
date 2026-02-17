using Google.Protobuf.Reflection;
using OpenCvSharp;
using OpenCvSharp.ML;
using Sam2Sharp.Modeling;
using Sam2Sharp.Utils;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using WinRT;
using static Sam2Sharp.Utils.Classes;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
namespace SAM2Sharp
{
    public class SAM2ImagePredictor
    {
        public readonly Sam2Base model;
        public readonly Device device;
        private readonly ScalarType dtype;
        private long[] original_size = null;
        //private float scaleFactor = 1.0f;
        public readonly SAM2Transforms transforms;
        private bool is_image_set;
        private Dictionary<string, object> _features;
        private List<(int, int)> orig_hw;
        private bool _is_batch;
        private readonly float mask_threshold;
        private readonly float max_sprinkle_area;
        private readonly List<(int, int)> _bb_feat_sizes = new()
        {
           // (256,256), (128,128), (64,64)//image_size=1024
            (128,128), (64,64), (32,32)//image_size=512
        };
        float max_hole_area;
        public readonly double[] pixel_mean = [0.485f, 0.456f, 0.406f];
        public readonly double[] pixel_std = [0.229f, 0.224f, 0.225f];

        //private readonly float[] pixel_mean = new float[] { 123.675f, 116.28f, 103.53f };
        //private readonly float[] pixel_std = new float[] { 58.395f, 57.12f, 57.375f };
        //private readonly double[] pixel_mean = [ 123.675, 116.28, 103.53 ];
        //private readonly double[] pixel_std = [ 58.395, 57.12, 57.375 ];


        public SAM2ImagePredictor(Sam2Base samModel, float mask_threshold = 0.0f, float max_hole_area = 0.0f, float max_sprinkle_area = 0.0f, params object[] kwargs)
        {
            model = samModel;
            mask_threshold = mask_threshold;
            transforms = new SAM2Transforms(
                resolution: samModel.image_size,
                mask_threshold: mask_threshold,
                max_hole_area: max_hole_area,
                max_sprinkle_area: max_sprinkle_area,
                device: model.device,
                dtype: model.dtype
            );
            _bb_feat_sizes = [(model.image_size / 4, model.image_size / 4), (model.image_size / 8, model.image_size / 8), (model.image_size / 16, model.image_size / 16)];

            //if (samModel.image_size == 1024)
            //    _bb_feat_sizes = [(256, 256), (128, 128), (64, 64)];
            //else if (samModel.image_size == 512)
            //    _bb_feat_sizes = [(128, 128), (64, 64), (32, 32)];
            //else
            //    _bb_feat_sizes = [(64, 64), (32, 32), (16, 16)];

            is_image_set = false;
            _features = null;
            orig_hw = null;
            _is_batch = false;
        }

        //public static SAM2ImagePredictor FromPretrained(string modelId, params object[] kwargs)
        //{
        //    var samModel = BuildSam2Hf(modelId, kwargs);
        //    return new SAM2ImagePredictor(samModel, 0.0f);
        //}
        /// <summary>
        /// Init Predictor, you don't have to choose Vit-b, Vit-l or Vit-H. It will be auto selected when loading model.
        /// </summary>
        /// <param name="checkpointPath">Checkpoint Path</param>
        /// <param name="device">Sam2 Device, it's CPU or Cuda.</param>
        public SAM2ImagePredictor(string checkpointPath, Device device, ScalarType dtype)
        {
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager(100);
            this.device = device;
            this.dtype = dtype;
            model = BuildSam2.BuildSam2Model(checkpointPath, this.device, this.dtype);
            //mask_threshold = mask_threshold;
            //max_sprinkle_area = this
            transforms = new SAM2Transforms(
                resolution: model.image_size,
                device: device,
                dtype: dtype,
                mask_threshold: mask_threshold,
                max_hole_area: max_hole_area,
                max_sprinkle_area: max_sprinkle_area
            );
            _bb_feat_sizes = [(model.image_size / 4, model.image_size / 4), (model.image_size / 8, model.image_size / 8), (model.image_size / 16, model.image_size / 16)];

        }

        private static Sam2Base BuildSam2Hf(string modelId, params object[] kwargs)
        {
            throw new NotImplementedException("BuildSam2Hf is not implemented");
        }
      
        [TorchFunction("set_image")]
        public void set_image(string img_path)
        {
            if (_features != null && _features.Count > 0)
            {
                foreach (var f in _features)
                {
                    if (f.Value is Tensor tensor)
                        tensor.Dispose();
                    if (f.Value is Tensor[] tensor1)
                    {
                        foreach (var t in tensor1)
                            t.Dispose();
                    }
                    if (f.Value is List<Tensor> tensor2)
                    {
                        foreach (var t in tensor2)
                            t.Dispose();
                    }
                }
                _features.Clear();
                _features = null;
            }
            GC.Collect();
            using var _ = no_grad();
            ResetPredictor();
            if(orig_hw==null||orig_hw.Count==0)
                orig_hw = new List<(int, int)>();
            var inputImage = transforms.mat_tensor(img_path);
            orig_hw.Add(inputImage.Item2);
        //    inputImage = inputImage.to(dtype,device);
        //    input_size = transforms.input_size;
        //    scaleFactor = transforms.scaleFactor;
            if (inputImage.Item1.shape.Length != 4 || inputImage.Item1.shape[1] != 3)
                throw new ArgumentException($"input_image must be of size 1x3xHxW, got {inputImage.Item1.shape}");
            
            var backboneOut = model.forward_image(inputImage.Item1);

            // 显式声明析构变量类型
            (object, List<Tensor>, object, object) backboneFeatures = model._prepare_backbone_features(backboneOut);
            List<Tensor> visionFeats = backboneFeatures.Item2;

            if (model.directly_add_no_mem_embed)
                visionFeats[visionFeats.Count - 1] = visionFeats[visionFeats.Count - 1] + model.no_mem_embed;

            var feats = new List<Tensor>();
            for (int i = 0; i < visionFeats.Count; i++)
            {
                var feat = visionFeats[visionFeats.Count - 1 - i];
                var featSize = _bb_feat_sizes[_bb_feat_sizes.Count - 1 - i];
                feats.Add(feat.permute(1, 2, 0).view(1, -1, featSize.Item1, featSize.Item2));
            }
            feats.Reverse();

            _features = new Dictionary<string, object>
            {
                { "image_embed", feats.Last() },
                { "high_res_feats", feats.Take(feats.Count - 1).ToList() }
            };

            //input_features = (Tensor)backboneOut["vision_features"];
            ////model.image_embeddings = input_features;
            ////var dd = input_features.to(ScalarType.Float32);
            ////var array = dd.data<float>().ToArray();

            //// 显式声明析构变量类型
            //(object, List<Tensor>, object, object) backboneFeatures = model._prepare_backbone_features(backboneOut);
            //List<Tensor> visionFeats = backboneFeatures.Item2;

            //if (model.directly_add_no_mem_embed)
            //    visionFeats[visionFeats.Count - 1] = visionFeats[visionFeats.Count - 1] + model.no_mem_embed;

            //var feats = new List<Tensor>();
            //for (int i = 0; i < visionFeats.Count; i++)
            //{
            //    var feat = visionFeats[visionFeats.Count - 1 - i];
            //    var featSize = _bb_feat_sizes[_bb_feat_sizes.Count - 1 - i];
            //    feats.Add(feat.permute(1, 2, 0).view(1, -1, featSize.Item1, featSize.Item2));
            //}
            //feats.Reverse();

            //_features = new Dictionary<string, object>
            //{
            //    { "image_embed", feats.Last() },
            //    { "high_res_feats", feats.Take(feats.Count - 1).ToArray() }
            //};
            is_image_set = true;
        }

        public void set_image(Mat image)
        {
            //if (_features != null && _features.Count > 0)
            //{
            //    foreach (var f in _features)
            //    {
            //        if (f.Value is Tensor tensor)
            //            tensor.Dispose();
            //        if (f.Value is Tensor[] tensor1)
            //        {
            //            foreach (var t in tensor1)
            //                t.Dispose();
            //        }
            //        if (f.Value is List<Tensor> tensor2)
            //        {
            //            foreach (var t in tensor2)
            //                t.Dispose();
            //        }
            //    }
            //    _features.Clear();
            //    _features = null;
            //}
            GC.Collect();
            long start = DateTime.Now.Ticks;
            using var _ = no_grad();
           // ResetPredictor();
            if (orig_hw == null || orig_hw.Count == 0)
                orig_hw = new List<(int, int)>();
            orig_hw.Add((image.Height, image.Width));
            var inputImage = transforms.mat_tensor(image);
            //    inputImage = inputImage.to(dtype,device);
            //    input_size = transforms.input_size;
            //    scaleFactor = transforms.scaleFactor;
            if (inputImage.shape.Length != 4 || inputImage.shape[1] != 3)
                throw new ArgumentException($"input_image must be of size 1x3xHxW, got {inputImage.shape}");

            var backboneOut = model.forward_image(inputImage);
            long end = DateTime.Now.Ticks;
            long GIelapsedMs = (end - start) / 10000;

            // 显式声明析构变量类型
            (object, List<Tensor>, object, object) backboneFeatures = model._prepare_backbone_features(backboneOut);
            List<Tensor> visionFeats = backboneFeatures.Item2;

            if (model.directly_add_no_mem_embed)
                visionFeats[visionFeats.Count - 1] = visionFeats[visionFeats.Count - 1] + model.no_mem_embed;

            var feats = new List<Tensor>();
            for (int i = 0; i < visionFeats.Count; i++)
            {
                var feat = visionFeats[visionFeats.Count - 1 - i];
                var featSize = _bb_feat_sizes[_bb_feat_sizes.Count - 1 - i];
                feats.Add(feat.permute(1, 2, 0).view(1, -1, featSize.Item1, featSize.Item2));
            }
            feats.Reverse();

            _features = new Dictionary<string, object>
            {
                { "image_embed", feats.Last() },
                { "high_res_feats", feats.Take(feats.Count - 1).ToList() }
            };

            //input_features = (Tensor)backboneOut["vision_features"];
            ////model.image_embeddings = input_features;
            ////var dd = input_features.to(ScalarType.Float32);
            ////var array = dd.data<float>().ToArray();

            //// 显式声明析构变量类型
            //(object, List<Tensor>, object, object) backboneFeatures = model._prepare_backbone_features(backboneOut);
            //List<Tensor> visionFeats = backboneFeatures.Item2;

            //if (model.directly_add_no_mem_embed)
            //    visionFeats[visionFeats.Count - 1] = visionFeats[visionFeats.Count - 1] + model.no_mem_embed;

            //var feats = new List<Tensor>();
            //for (int i = 0; i < visionFeats.Count; i++)
            //{
            //    var feat = visionFeats[visionFeats.Count - 1 - i];
            //    var featSize = _bb_feat_sizes[_bb_feat_sizes.Count - 1 - i];
            //    feats.Add(feat.permute(1, 2, 0).view(1, -1, featSize.Item1, featSize.Item2));
            //}
            //feats.Reverse();

            //_features = new Dictionary<string, object>
            //{
            //    { "image_embed", feats.Last() },
            //    { "high_res_feats", feats.Take(feats.Count - 1).ToArray() }
            //};
            is_image_set = true;
        }

        Tensor input_features;
        public void set_image_batch(List<string> imageList)
        {
            using var _ = no_grad();
            ResetPredictor();
            if (orig_hw == null || orig_hw.Count == 0)
                orig_hw = new List<(int, int)>();
            var imgBatch = transforms.forward_batch(imageList);
            var tenBatch = stack(imgBatch.Item1.ToArray(), dim:0);
            tenBatch = tenBatch.to(dtype, device);
            var batchSize = (int)tenBatch.shape[0];
            orig_hw = imgBatch.Item2;
            if (tenBatch.shape.Length != 4 || tenBatch.shape[1] != 3)
                throw new ArgumentException($"img_batch must be of size Bx3xHxW, got {tenBatch.shape}");

            var backboneOut = model.forward_image(tenBatch);

            // 显式声明析构变量类型
            (object, List<Tensor>, object, object) backboneFeatures = model._prepare_backbone_features(backboneOut);
            List<Tensor> visionFeats = backboneFeatures.Item2;

            if (model.directly_add_no_mem_embed)
                visionFeats[visionFeats.Count - 1] = visionFeats[visionFeats.Count - 1] + model.no_mem_embed;

            var feats = new List<Tensor>();
            for (int i = 0; i < visionFeats.Count; i++)
            {
                var feat = visionFeats[visionFeats.Count - 1 - i];
                var featSize = _bb_feat_sizes[_bb_feat_sizes.Count - 1 - i];
                feats.Add(feat.permute(1, 2, 0).view(batchSize, -1, featSize.Item1, featSize.Item2));
            }
            feats.Reverse();

            _features = new Dictionary<string, object>
            {
                { "image_embed", feats.Last() },
                { "high_res_feats", feats.Take(feats.Count - 1).ToList() }
            };

            is_image_set = true;
            _is_batch = true;
        }
        public List<List<PredictOutput>> Predict_Batch(List<List<SamPoint>> point_batch, bool multimask_output = true, bool return_logits = false, bool normalize_coords = true)
        {
            using var _ = no_grad();

            /*This function is very similar to predict(...), however it is used for batched mode, when the model is expected to generate predictions on multiple images.
            It returns a tuple of lists of masks, ious, and low_res_masks_logits.
            */
            if (!is_image_set || !_is_batch)
                throw new Exception("An image must be set with .set_image_batch(...) before mask prediction.");
            int num_images = (int)(_features["image_embed"] as Tensor).shape[0];
            Tensor[] all_masks = new Tensor[num_images], all_ious = new Tensor[num_images], all_low_res_masks = new Tensor[num_images];
            List<List<PredictOutput>> all_predictOutputs = new List<List<PredictOutput>>();
            for (int img_idx = 0; img_idx < num_images; img_idx++)
            {
                // Transform input prompts
                //(Tensor unnorm_coords, Tensor labels, Tensor mask_inputTensor, Tensor unnorm_box) = transforms.points_to_Tensor(point_batch[img_idx], orig_hw[img_idx], normalize_coords = true);

                var (mask_inputTensor, unnorm_coords, labels, unnorm_box) = _prep_prompts(point_batch[img_idx],null, null, normalize_coords = true,img_idx=img_idx);
                //var dd = unnorm_coords.to(ScalarType.Float32);
                //var array = dd.data<float>().ToArray();

                //var dd1 = unnorm_coords1.to(ScalarType.Float32);
                //var array1 = dd1.data<float>().ToArray();

                var (masks, iou_predictions, low_res_masks) = PredictCore(unnorm_coords, labels, unnorm_box, null, multimask_output, return_logits = return_logits,img_idx=img_idx);

                List<PredictOutput> predictOutputs = new List<PredictOutput>();
                var Masks = masks.squeeze(dim: 0);
                var Iou_predictions = iou_predictions.squeeze(dim: 0);
                for (int i = 0; i < Masks.shape[0]; i++)
                {
                    bool[,] maskArray = new bool[Masks.shape[2], Masks.shape[1]];
                    var data = Masks.transpose(1, 2)[i].data<bool>().ToArray();
                    Buffer.BlockCopy(data, 0, maskArray, 0, data.Length * sizeof(bool));

                    predictOutputs.Add(new PredictOutput
                    {
                        Mask = maskArray,
                        Precision = Iou_predictions[i].ToSingle(),
                    });
                }
                masks.Dispose();
                iou_predictions.Dispose();
                low_res_masks.Dispose();
                all_predictOutputs.Add(predictOutputs);
            }
            GC.Collect();
            return all_predictOutputs;
        }

        public (Tensor, Tensor, Tensor) Predict0(Tensor point_coords = null, Tensor point_labels = null, Tensor box = null, Tensor mask_input = null, bool multimask_output = false, bool return_logits = false, bool normalize_coords = true)
        {
            if (!is_image_set)
                throw new InvalidOperationException("An image must be set with .set_image(...) before mask prediction.");

            //var (mask_inputTensor, unnorm_coords, labels, unnorm_box) = _prep_prompts(
            //    point_coords, point_labels, box, mask_input, normalize_coords);

            // var (masks, iou_predictions, low_res_masks) = PredictCore(
            //unnorm_coords,
            //labels,
            //unnorm_box,
            //mask_input,
            //multimask_output,
            //return_logits = return_logits);
            var (masks, iou_predictions, low_res_masks) = PredictCore(point_coords, point_labels, box, mask_input, multimask_output, return_logits = return_logits);
            return (masks, iou_predictions, low_res_masks);

            //// Placeholder simple prediction: return empty tensors
            //var device = Device;
            //var empty = torch.zeros(new long[] { 1, 1, 1, 1 }, device: device);
            //return (empty, empty, empty);
        }


        public List<PredictOutput> Predict1(List<SamPoint> points = null)
        {
            using var _ = no_grad();
            using var __ = NewDisposeScope();
            model.eval();


            Tensor pointsTensor = null;
            Tensor labelsTensor = null;

            if (points is not null)
                (pointsTensor, labelsTensor,Tensor mask_inputTensor,  Tensor unnorm_box) = transforms.points_to_Tensor(points, null, null, true, orig_hw[0]);           

            BatchedInput batchedInput = new BatchedInput
            {
                Point_coords = pointsTensor.to(dtype,device),
                Point_labels = labelsTensor.to(dtype, device),
                orig_hw = (orig_hw[0].Item1, orig_hw[0].Item2),
                //Input_size = input_size
            };
            
            BatchedOutput output = model.forward(batchedInput, false);
            List<PredictOutput> predictOutputs = new List<PredictOutput>();
         //   output.Masks.squeeze(dim: 0);
            for (int i = 0; i < output.Masks.shape[1]; i++)
            {
                bool[,] maskArray = new bool[output.Masks.shape[3], output.Masks.shape[2]];
                var data = output.Masks.transpose(2, 3)[i].data<bool>().ToArray();
                Buffer.BlockCopy(data, 0, maskArray, 0, data.Length * sizeof(bool));

                predictOutputs.Add(new PredictOutput
                {
                    Mask = maskArray,
                    Precision = output.Iou_predictions[i].ToSingle(),
                });
            }
            //var masks = output.Masks;
            //var data = masks.transpose(2, 3).data<bool>().ToArray();
            //int len = (int)(masks.shape[3] * masks.shape[2]);
            //int k = (int)masks.shape[1];
            //for (int i = 0; i < k; i++)
            //{
            //    bool[,] maskArray = new bool[masks.shape[3], masks.shape[2]];
            //    Buffer.BlockCopy(data, i * len, maskArray, 0, maskArray.Length * sizeof(bool));

            //    float sc = output.Iou_predictions[i].ToSingle();

            //    predictOutputs.Add(new PredictOutput
            //    {
            //        Mask = maskArray,
            //        Precision = sc
            //    });
            //}
            output.Dispose();
            GC.Collect();
            return predictOutputs;
        }
        public List<PredictOutput> Predict((Tensor unnorm_coords, Tensor labels) point_labels)
        {
            using var _ = no_grad();
            using var __ = NewDisposeScope();
            model.eval();

            if (!is_image_set)
                throw new InvalidOperationException("An image must be set with .set_image(...) before mask prediction.");


            //(Tensor unnorm_coords, Tensor labels,Tensor mask_inputTensor,Tensor unnorm_box) = transforms.points_to_Tensor(points, orig_hw[0], normalize_coords=true);

            //var (mask_inputTensor, unnorm_coords, labels, unnorm_box) = _prep_prompts(points, box, mask_input, normalize_coords);
            //long end = DateTime.Now.Ticks;
            //long GIelapsedMs = (end - start) / TimeSpan.TicksPerMillisecond;
            //Debug.WriteLine($"Mask Decoder Time: {GIelapsedMs} ms");
            //var dd = unnorm_coords.to(ScalarType.Float32);
            //var array = dd.data<float>().ToArray();

            //var dd1 = unnorm_coords1.to(ScalarType.Float32);
            //var array1 = dd1.data<float>().ToArray();

            var (masks, iou_predictions, low_res_masks) = PredictCore(point_labels.unnorm_coords, point_labels.labels, null, null, false, return_logits:false);
            //var (masks, iou_predictions, low_res_masks) = PredictCore(point_coords, point_labels, box,mask_input, multimask_output,return_logits = return_logits);

            List<PredictOutput> predictOutputs = new List<PredictOutput>();
         //   var Masks = masks.squeeze(dim: 0);
            long start = DateTime.Now.Ticks;
         //   Masks = /*Masks.device == torch.CPU ? Masks :*/ Masks.to(torch.CPU);
            var Iou_predictions = iou_predictions.squeeze(dim: 0);
    //        for (int i = 0; i < masks.shape[0]; i++)
    //        {
    ////            bool[] flatBoolArray = Masks[i].ToArray<bool>();
    //            //start = DateTime.Now.Ticks;
    //            bool[,] maskArray = new bool[masks.shape[3], masks.shape[2]];
    //            var data = masks.transpose(2, 3)[i].data<bool>().ToArray();
    //            Buffer.BlockCopy(data, 0, maskArray, 0, data.Length * sizeof(bool));
    //            predictOutputs.Add(new PredictOutput
    //            {
    //                Mask = maskArray,
    //                Precision = Iou_predictions[i].ToSingle(),
    //            });
    //        }

            //List<PredictOutput> predictOutputs = new List<PredictOutput>();
            //long start = DateTime.Now.Ticks;
            for (int i = 0; i < masks.shape[0]; i++)
            {
                bool[,] maskArray = new bool[masks.shape[3], masks.shape[2]];
                var data = masks.transpose(2, 3)[i].data<bool>().ToArray();
                Buffer.BlockCopy(data, 0, maskArray, 0, data.Length * sizeof(bool));
                predictOutputs.Add(new PredictOutput
                {
                    Mask = maskArray,
                    Precision = iou_predictions[i].ToSingle(),
                });
            }
            long end = DateTime.Now.Ticks;
            long GIelapsedMs = (end - start) / TimeSpan.TicksPerMillisecond;
            Debug.WriteLine($"Prompt + Mask Decoder Time000: {GIelapsedMs} ms; " + masks.shape[0]);

            masks.Dispose();
            iou_predictions.Dispose();
            low_res_masks.Dispose();
            GC.Collect();
            return predictOutputs;

            // return (masks, iou_predictions, low_res_masks);

            //// Placeholder simple prediction: return empty tensors
            //var device = Device;
            //var empty = torch.zeros(new long[] { 1, 1, 1, 1 }, device: device);
            //return (empty, empty, empty);
        }

        public /*(Tensor, Tensor, Tensor)*/ List<PredictOutput> Predict(List<SamPoint> points = null,Tensor box = null,Tensor mask_input = null, bool multimask_output = false, bool return_logits = false, bool normalize_coords = true)
        {
            using var _ = no_grad();
            using var __ = NewDisposeScope();
            model.eval();

            if (!is_image_set)
                throw new InvalidOperationException("An image must be set with .set_image(...) before mask prediction.");
            long start = DateTime.Now.Ticks;


            //(Tensor unnorm_coords, Tensor labels,Tensor mask_inputTensor,Tensor unnorm_box) = transforms.points_to_Tensor(points, orig_hw[0], normalize_coords=true);

            var (mask_inputTensor, unnorm_coords, labels, unnorm_box) = _prep_prompts(points, box, mask_input, normalize_coords);
            long end = DateTime.Now.Ticks;
            long GIelapsedMs = (end - start) / TimeSpan.TicksPerMillisecond;
            Debug.WriteLine($"Prep_Prompts Time: {GIelapsedMs} ms");
            //var dd = unnorm_coords.to(ScalarType.Float32);
            //var array = dd.data<float>().ToArray();

            //var dd1 = unnorm_coords1.to(ScalarType.Float32);
            //var array1 = dd1.data<float>().ToArray();

            var (masks, iou_predictions, low_res_masks) = PredictCore(unnorm_coords,labels,unnorm_box,mask_input, multimask_output,return_logits = return_logits);
            //var (masks, iou_predictions, low_res_masks) = PredictCore(point_coords, point_labels, box,mask_input, multimask_output,return_logits = return_logits);

            List<PredictOutput> predictOutputs = new List<PredictOutput>();
            //var Masks = masks.squeeze(dim: 0);
            var Iou_predictions = iou_predictions.squeeze(dim: 0);
            for (int i = 0; i < masks.shape[0]; i++)
            {
                torch.cuda.synchronize();
                torch.cuda.synchronize();
                bool[,] maskArray = new bool[masks.shape[3], masks.shape[2]];
                var data = masks.transpose(2, 3)[i].data<bool>().ToArray();
                Buffer.BlockCopy(data, 0, maskArray, 0, data.Length * sizeof(bool));
                predictOutputs.Add(new PredictOutput
                {
                    Mask = maskArray,
                    Precision = iou_predictions[i].ToSingle(),
                });
            }
            masks.Dispose();
            iou_predictions.Dispose();
            low_res_masks.Dispose();
            GC.Collect();
            return predictOutputs;


            // return (masks, iou_predictions, low_res_masks);

            //// Placeholder simple prediction: return empty tensors
            //var device = Device;
            //var empty = torch.zeros(new long[] { 1, 1, 1, 1 }, device: device);
            //return (empty, empty, empty);
        }
        
        private (Tensor, Tensor, Tensor, Tensor) _prep_prompts(List<SamPoint> points, Tensor box, Tensor maskLogits, bool normalizeCoords, int imgIdx = 0)
        {
            Tensor unnorm_coords = null, pointcoords=null, labels = null, unnorm_box = null, mask_input = null;
            if(points is not null)
            {
                float[] coordsArray = points.SelectMany(p => new float[] { p.X, p.Y }).ToArray();
                int[] labelsArray = points.Select(p => p.Label.HasValue && p.Label.Value ? 1 : 0).ToArray();
                // 步骤2：将数组转为Torch张量，并重塑为[N, 2]（N是点的数量）
                pointcoords = torch.tensor(
                    coordsArray,
                    dtype: dtype,
                    device: device
                ).reshape(-1, 2);  // -1表示自动计算维度，确保形状为[N, 2]
                labels = torch.tensor(
                    labelsArray,
                    dtype: dtype,
                    device: device
                );
                //var coords = pointcoords.clone();
                //if (normalizeCoords)
                {
                    //var xSlice = unnorm_coords.index_select(-1, tensor(0));  // 取出所有X坐标
                    //unnorm_coords.index_put_(new TensorIndex[] { TensorIndex.Ellipsis, tensor(0) }, xSlice / orig_hw[0].Item2);

                    //// 步骤3：对最后一维的第1个元素（Y坐标）除以h
                    //var ySlice = unnorm_coords.index_select(-1, tensor(1));  // 取出所有Y坐标
                    //unnorm_coords.index_put_(new TensorIndex[] { TensorIndex.Ellipsis, tensor(1) },  ySlice / orig_hw[0].Item2);
                    pointcoords[.., 0] = pointcoords[.., 0] / orig_hw[0].Item2;
                    pointcoords[.., 1] = pointcoords[.., 1] / orig_hw[0].Item1;
                  //  pointcoords = coords;
                }
                //pointcoords = pointcoords.clone() * model.image_size;
                pointcoords = pointcoords * model.image_size;
                pointcoords = pointcoords[TensorIndex.None, ..];
                labels = labels[TensorIndex.None, ..];
                //unnorm_coords.unsqueeze(0);
                //labels.unsqueeze(0);
            }
            return (mask_input, pointcoords, labels, unnorm_box);
        }

        private (Tensor, Tensor, Tensor) PredictCore(Tensor point_coords = null,Tensor point_labels = null,Tensor boxes = null, Tensor mask_input = null, bool multimask_output = false, bool return_logits = false,int img_idx = 0)
         {
            using var _ = no_grad();
            // 1. 拼接点坐标和标签
            (Tensor, Tensor)? concat_points = null;
            if (point_coords is not null)
            {
                //point_coords.to(dtype, device);
                //point_labels.to(dtype, device);
                concat_points = (point_coords, point_labels);
            }

            // 2. 处理框输入并合并到concat_points
            if (boxes is not null && boxes.numel()!=0)
            {
                // 重塑框坐标为[N, 2, 2]（每个框包含左上/右下两个点）
                var box_coords = boxes.reshape(new long[] { -1, 2, 2 });

                // 创建框标签 [2,3]（SAM约定：2=框左上，3=框右下）
                var box_labels = tensor(new int[,] { { 2, 3 } }, ScalarType.Int32, boxes.device)
                    .repeat(new long[] { boxes.size(0), 1 });

                //// 合并框和点输入（框在前，点在后）
                //if (concat_points.HasValue)
                //{
                //    var concat_coords = torch.cat(new[] { box_coords, concat_points.Value.Item1 }, dim: 1);
                //    var concat_labels = torch.cat(new[] { box_labels, concat_points.Value.Item2 }, dim: 1);
                //    concat_points = (concat_coords, concat_labels);
                //}
                //else
                //{
                //    concat_points = (box_coords, box_labels);
                //}
            }
            //var dd = point_coords.to(ScalarType.Float32);
            //var array = dd.data<float>().ToArray();

            // 3. 编码Prompt（调用SAM Prompt Encoder）
            (Tensor sparse_embeddings, Tensor dense_embeddings) = model.sam_prompt_encoder.forward(points: concat_points,boxes: boxes,masks: mask_input);
            //long end = DateTime.Now.Ticks;
            //long GIelapsedMs = (end - start) / TimeSpan.TicksPerMillisecond;
            //Debug.WriteLine($"Prompt Decoder Time: {GIelapsedMs} ms");

            // 4. 判断是否为批量模式（多对象预测）
            bool batched_mode = concat_points.HasValue && concat_points.Value.Item1.shape[0] > 1;

            // 5. 提取高分辨率特征（添加batch维度）
            var high_res_features = new List<Tensor>();
            //var features = _features["high_res_feats"] as Tensor[];
            var features = (_features["high_res_feats"]) as List<Tensor>;
            foreach (var feat_level in features)
            {
                high_res_features.Add(feat_level[img_idx].unsqueeze(0));
            }
            var input_features = ((Tensor)_features["image_embed"])[img_idx].unsqueeze(0).to(dtype, device);
            //var dd = input_features.to(ScalarType.Float32);
            //var array = dd.data<float>().ToArray();

            // 6. 预测掩码（调用SAM Mask Decoder）
            var (low_res_masks, iou_predictions, _, _) = model.sam_mask_decoder.forward(
                (input_features,
                 model.sam_prompt_encoder.get_dense_pe(),
                sparse_embeddings,
                dense_embeddings,
                multimask_output,
                batched_mode,
                high_res_features));
            // 7. 将掩码上采样到原始图像分辨率
            var masks = transforms.postprocess_masks(low_res_masks,orig_hw[img_idx]);

           // masks = masks.squeeze(0);

            //long start = DateTime.Now.Ticks;
            //torch.cuda.synchronize();
            //var masks_np = masks.@float().detach().cpu();
            //torch.cuda.synchronize();
            //long end1 = DateTime.Now.Ticks;
            //long GIelapsedMs = (end1 - start) / TimeSpan.TicksPerMillisecond;
            //Debug.WriteLine($"ToCPU Time1: {GIelapsedMs} ms");


            //var Maskcpu = torch.empty(masks.shape[0], masks.shape[1], masks.shape[2], dtype:torch.bfloat16, device:CPU).pin_memory();
            //torch.cuda.synchronize();
            //Maskcpu.copy_(masks);
            //torch.cuda.synchronize();
            //masks = Maskcpu;

            //long end = DateTime.Now.Ticks;
            //GIelapsedMs = (end - end1) / TimeSpan.TicksPerMillisecond;
            //Debug.WriteLine($"ToCPU Time2: {GIelapsedMs} ms");


            //dd = masks.to(ScalarType.Float32);
            //array = dd.data<float>().ToArray();


            // 8. 限制低分辨率掩码的数值范围（防止溢出）
            low_res_masks = torch.clamp(low_res_masks, -32.0f, 32.0f);

            // 9. 转换为二值掩码（如果不返回logits）
            if (!return_logits)
            {
                masks = masks > 0.0;
            }
            //var array = masks.data<bool>().ToArray();

            sparse_embeddings.Dispose();
            dense_embeddings.Dispose();
            //long end = DateTime.Now.Ticks;
            //long GIelapsedMs = (end - start) / TimeSpan.TicksPerMillisecond;
            //Debug.WriteLine($"Prompt + Mask Decoder Time: {GIelapsedMs} ms");

            //foreach (var f in high_res_features)
            //    f.Dispose();
            //foreach (var f in features)
            //    f.Dispose();
            //foreach (var f in _features)
            //{
            //    if (f.Value is Tensor tensor)
            //        tensor.Dispose();
            //    if (f.Value is Tensor[] tensor1)
            //    {
            //        foreach (var t in tensor1)
            //            t.Dispose();
            //    }
            //    if (f.Value is List<Tensor> tensor2)
            //    {
            //        foreach (var t in tensor2)
            //            t.Dispose();
            //    }
            //}
            //_features.Clear();
            //_features = null;
            //features.Clear();
            //features = null;
            //foreach (var f in _features)
            //    f.Dispose();
            ////var features = _features["high_res_feats"] as Tensor[];
            //var features = (_features["high_res_feats"]) as List<Tensor>;
            //foreach (var feat_level in features)
            //{
            //    high_res_features.Add(feat_level[img_idx].unsqueeze(0));
            //}
            //input_features.Dispose();
            //GC.Collect();
            return (masks.MoveToOuterDisposeScope(), iou_predictions.MoveToOuterDisposeScope(), low_res_masks.MoveToOuterDisposeScope());
        }

        public Tensor GetImageEmbedding()
        {
            if (!is_image_set)
                throw new InvalidOperationException("An image must be set with .set_image(...) to generate an embedding.");

            if (_features == null)
                throw new InvalidOperationException("Features must exist if an image has been set.");

            return _features["image_embed"] as Tensor;
        }

        // 替换原有 Device 属性，直接返回 model 的 device 字段
        public Device Device => model.device;

        public SAM2Transforms Transforms => transforms;

        public void ResetPredictor()
        {
            is_image_set = false;
            //_features?.ToList().ForEach(kv => kv.Value.Dispose());
            _features = null;
            orig_hw = null;
            _is_batch = false;
        }
    }
}