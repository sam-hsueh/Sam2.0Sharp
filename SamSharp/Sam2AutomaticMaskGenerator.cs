using OpenCvSharp;
using OpenCvSharp.ML;
using Sam2Sharp.Modeling;
using Sam2Sharp.Utils;
using System.Threading.Tasks;
using TorchSharp;
using static Sam2Sharp.Utils.AMG;
using static Tensorboard.ApiDef.Types;
using static TorchSharp.torch;

namespace SAM2Sharp
{    

    public class SAM2AutomaticMaskGenerator
    {
        private SAM2ImagePredictor predictor;
        private float mask_threshold;
        private bool multimask_output;
        private bool use_m2m;
        private string output_mode = "binary_mask";
        //private int points_per_batch;
        //private float pred_iou_thresh;
        //private float stability_score_thresh;
        //private float stability_score_offset;
        //private float box_nms_thresh;
        //private int crop_n_layers;
        //private float crop_nms_thresh;
        //private float crop_overlap_ratio;
        //private int crop_n_points_downscale_factor;
        //private int min_mask_region_area;
        //private List<Tensor> point_grids;
        private Sam2Base model;

        private readonly int points_per_side = 32;
        private readonly int points_per_batch = 64;
        private readonly float pred_iou_thresh = 0.8f;
        private readonly float stability_score_thresh = 0.95f;
        private readonly float stability_score_offset = 1.0f;
        private readonly float box_nms_thresh = 0.7f;
        private readonly int crop_n_layers = 2;
        private readonly float crop_nms_thresh = 0.7f;
        private readonly float crop_overlap_ratio = 512.0f / 1500;
        private readonly int crop_n_points_downscale_factor = 1;
        private readonly List<Tensor> point_grids;
        private readonly int min_mask_region_area = 0;
        //private readonly OutputMode output_mode = OutputMode.BinaryMask;

        //private readonly Sam model;
        private readonly Device device;
        private readonly ScalarType dtype;



        public SAM2AutomaticMaskGenerator(
            Sam2Base model,
            int? points_per_side = 32,
            int points_per_batch = 64,
            float pred_iou_thresh = 0.8f,
            float stability_score_thresh = 0.95f,
            float stability_score_offset = 1.0f,
            float mask_threshold = 0.0f,
            float box_nms_thresh = 0.7f,
            int crop_n_layers = 0,
            float crop_nms_thresh = 0.7f,
            float crop_overlap_ratio = 512f / 1500f,
            int crop_n_points_downscale_factor = 1,
            List<Tensor> point_grids = null,
            int min_mask_region_area = 0,
            string output_mode = "binary_mask",
            bool use_m2m = false,
            bool multimask_output = true)
        {
            if (points_per_side == null && point_grids == null)
                throw new ArgumentException("Exactly one of points_per_side or point_grid must be provided.");

            if (points_per_side != null)
            {
                this.point_grids = build_all_layer_point_grids(
                    points_per_side.Value,
                    crop_n_layers,
                    crop_n_points_downscale_factor);
            }
            else
            {
                this.point_grids = point_grids;
            }

            var validOutputModes = new[] { "binary_mask", "uncompressed_rle", "coco_rle" };
            if (!validOutputModes.Contains(output_mode))
                throw new ArgumentException($"Unknown output_mode {output_mode}.");

            this.predictor = new SAM2ImagePredictor(model, min_mask_region_area, min_mask_region_area);
            this.model = model;
            this.points_per_batch = points_per_batch;
            this.pred_iou_thresh = pred_iou_thresh;
            this.stability_score_thresh = stability_score_thresh;
            this.stability_score_offset = stability_score_offset;
            this.mask_threshold = mask_threshold;
            this.box_nms_thresh = box_nms_thresh;
            this.crop_n_layers = crop_n_layers;
            this.crop_nms_thresh = crop_nms_thresh;
            this.crop_overlap_ratio = crop_overlap_ratio;
            this.crop_n_points_downscale_factor = crop_n_points_downscale_factor;
            this.min_mask_region_area = min_mask_region_area;
            this.output_mode = output_mode;
            this.use_m2m = use_m2m;
            this.multimask_output = multimask_output;
        }

        public static SAM2AutomaticMaskGenerator FromPretrained(string model_id, Dictionary<string, object> kwargs = null)
        {
            //var samModel = BuildSam2Hf(model_id, kwargs);
            //return new SAM2AutomaticMaskGenerator(samModel);// From sam2_video_predictor_legacy,In Build_Sam
            return null;
        }

        public List<Dictionary<string, object>> generate(Tensor image)
        {
            using (var scope = torch.no_grad())
            {
                var maskData = generate_masks(image);

                var segmentations = new List<object>();
                if (output_mode == "coco_rle")
                {
                    //foreach (var rle in maskData.Rles)
                    //    segmentations.Add(coco_encode_rle(rle));
                }
                else if (output_mode == "binary_mask")
                {
                    //foreach (var rle in maskData.Rles)
                    //    segmentations.Add(rle_to_mask(rle));
                }
                else
                {
                    segmentations.AddRange(maskData.Rles.Cast<object>());
                }

                var annotations = new List<Dictionary<string, object>>();
                for (int i = 0; i < segmentations.Count; i++)
                {
                    var ann = new Dictionary<string, object>
                    {
                        {"segmentation", segmentations[i]},
                        {"area",area_from_rle(maskData.Rles[i])},
                        {"bbox", TensorToList(box_xyxy_to_xywh(maskData.boxes[i]))},
                        {"predicted_iou", maskData.iou_preds[i].item<float>()},
                        {"point_coords", new List<List<float>> { TensorToList(maskData.points[i]) }},
                        {"stability_score", maskData.stability_score[i].item<float>()},
                        {"crop_box", TensorToList(box_xyxy_to_xywh(maskData.crop_boxes[i]))}
                    };
                    annotations.Add(ann);
                }

                return annotations;
            }
        }

        private static List<float> TensorToList(Tensor t)
        {
            try
            {
                var arr = t.data<float>().ToArray();
                return arr.ToList();
            }
            catch
            {
                return new List<float>();
            }
        }

        private MaskData generate_masks(Tensor image)
        {
            using var _ = no_grad();
            using var __ = NewDisposeScope();
            model.eval();
            //Tensor imgTensor = Tools.ImageTools.GetTensorFromImage(image); long orig_w = imgTensor.shape[2];
            //long orig_h = imgTensor.shape[1];
            //float scaleFactor = Math.Min((float)maxImageSize / orig_w, (float)maxImageSize / orig_h);
            //int newW = (int)Math.Ceiling(orig_w * scaleFactor / 4) * 4;
            //int newH = (int)Math.Ceiling(orig_h * scaleFactor / 4) * 4;
            //imgTensor = torchvision.transforms.functional.resize(imgTensor, newH, newW).unsqueeze(0);
            //model.set_image(imgTensor, device, dtype);
            //long[] original_size = new long[] { orig_h, orig_w };
            //float xStart = (float)newW / points_per_batch;
            //float yStep = (float)newH / points_per_side;
            //float yStart = yStep / 2;

            //MaskData data = new MaskData();

            //List<BatchedOutput> outputs = new List<BatchedOutput>();
            //for (int y = 0; y < this.points_per_side / 2; y++)
            //{
            //    Tensor points = torch.zeros(new long[] { points_per_batch, 2 });
            //    points[..(this.points_per_batch / 2), 0] = torch.linspace(xStart, newW - xStart, points_per_batch / 2);
            //    points[(this.points_per_batch / 2).., 0] = torch.linspace(xStart, newW - xStart, points_per_batch / 2);
            //    points[..(this.points_per_batch / 2), 1] = yStart + yStep * (y * 2 + 0);
            //    points[(this.points_per_batch / 2).., 1] = yStart + yStep * (y * 2 + 1);
            //    Tensor labels = torch.ones(points.shape[0]);
            //    BatchedInput batched = new BatchedInput { Point_coords = points, Point_labels = labels, Original_size = original_size, Input_size = new long[] { newH, newW } };
            //    MaskData tempData = _process_batch(batched, crop_box, orig_w, orig_h);
            //    data.Concat(tempData);
            //    batched.Dispose();
            //    GC.Collect();
            //}

            //Tensor keep_by_nms = Amg.batched_nms(
            //    data.Boxes,
            //    data.IouPreds,
            //    torch.zeros_like(data.Boxes[.., 0]),  // categories
            //    iou_threshold: this.box_nms_thresh);

            //data.Filter(keep_by_nms);
            //List<PredictOutput> predictOutputs = new List<PredictOutput>();
            //for (int i = 0; i < data.IouPreds.shape[0]; i++)
            //{
            //    bool[,] maskArray = new bool[data.Masks.shape[2], data.Masks.shape[1]];
            //    var arrayData = data.Masks.transpose(2, 1)[i].data<bool>().ToArray();
            //    Buffer.BlockCopy(arrayData, 0, maskArray, 0, arrayData.Length * sizeof(bool));

            //    predictOutputs.Add(new PredictOutput
            //    {
            //        Mask = maskArray,
            //        Precision = data.IouPreds[i].ToSingle(),
            //    });

            //}
            //return predictOutputs;

            var origSize = ((int)image.shape[0], (int)image.shape[1]);
            var (cropboxes, layerIdxs) = generate_crop_boxes(origSize, crop_n_layers, crop_overlap_ratio);

            var data = new MaskData();
            foreach (var pair in cropboxes.Zip(layerIdxs, (c, l) => (c, l)))
            {
                var cropData = ProcessCrop(image, pair.c, pair.l, origSize);
                data.Cat(cropData);
            }

            // 移除不同裁剪之间的重复掩码
            if (cropboxes.Count > 1)
            {
                var scores = 1.0f / torchvision.ops.box_area(data.boxes);
                scores = scores.to(data.boxes[0].device);

                //var keep_by_nms = batched_nms(
                //    data.boxes,
                //    scores,
                //    data.boxes.Count,
                //    crop_nms_thresh
                //);
                Tensor keep_by_nms = batched_nms(
                  data.boxes,
                  data.iou_preds,
                  torch.zeros_like(data.boxes[.., 0]),  // categories
                  iou_threshold: this.box_nms_thresh);
                data.Filter(keep_by_nms);
            }
            return data;
        }

        private MaskData ProcessCrop(Tensor image, List<int> cropBox, int cropLayerIdx, (long, long) origSize)
        {
            // For compile: use whole image as cropped placeholder
            var croppedIm = image;
            var croppedImSize = ((int)croppedIm.shape[0], (int)croppedIm.shape[1]);

           // predictor.set_image(croppedIm);

            var data = new MaskData();
            foreach (var points in batch_iterator(points_per_batch, point_grids[cropLayerIdx]))
            {
                var batchData = _process_batch(points, croppedImSize, cropBox, ((int)origSize.Item1,(int)origSize.Item2), normalize: true);
                data.Cat(batchData);
            }

            predictor.ResetPredictor();

            // 移除当前裁剪内的重复掩码
            var keepByNms = batched_nms(
                data.boxes,
                data.iou_preds,
                data.boxes,
                box_nms_thresh
            );

            data.Filter(keepByNms);

            // 转换回原始图像坐标系
            data.boxes = uncrop_boxes_xyxy(data.boxes, cropBox);
            data.points = uncrop_points(data.points, cropBox);
            // 构建批量crop_boxes Tensor（形状：[N,4]，N=data.Rles.Count）
            using var cropBoxTensor = torch.tensor(cropBox, dtype: ScalarType.Float32);
            data.crop_boxes = cropBoxTensor
                .unsqueeze(0)          // [4] → [1,4]（增加批量维度）
                .repeat(data.Rles.Count, 1);

            return data;
        }
        /// <summary>
        /// 核心方法：_process_batch
        /// </summary>
        /// <param name="points">原Python：points: Tensor</param>
        /// <param name="im_size">原Python：im_size: Tuple[int, ...]</param>
        /// <param name="crop_box">原Python：crop_box: List[int]</param>
        /// <param name="orig_size">原Python：orig_size: Tuple[int, ...]</param>
        /// <param name="normalize">原Python：normalize=False</param>
        /// <returns>原Python：-> MaskData</returns>
        public MaskData _process_batch(Tensor points,(int, int) im_size,List<int> crop_box,(int, int) orig_size,bool normalize = false)
        {
            int orig_h = orig_size.Item1;
            int orig_w = orig_size.Item2;

            // 原Python：points = torch.as_tensor(...)
            points = torch.as_tensor(
                points,
                dtype: ScalarType.Float32,
                device: predictor.device
            );

            // 原Python：in_points = self.predictor._transforms.transform_coords(...)
            Tensor in_points = predictor.transforms.transform_coords(points,normalize: normalize, orig_hw: (im_size.Item1, im_size.Item2));

            // 原Python：in_labels = torch.ones(...)
            Tensor in_labels = torch.ones(
                in_points.shape[0],
                dtype: ScalarType.Int32,
                device: in_points.device
            );

            // 原Python：masks, iou_preds, low_res_masks = self.predictor._predict(...)
            var (masks, iou_preds, low_res_masks) = predictor.Predict0(
                in_points.unsqueeze(1),  // 对应[:, None, :]
                in_labels.unsqueeze(1),  // 对应[:, None]
                multimask_output: multimask_output,
                return_logits: true
            );

            // 原Python：data = MaskData(...)
            MaskData data = new MaskData
            {
                masks = masks.flatten(0, 1),
                iou_preds = iou_preds.flatten(0, 1),
                points = points.repeat_interleave(masks.shape[1], dim: 0),
                low_res_masks = low_res_masks.flatten(0, 1)
            };

            // 原Python：del masks
            masks.Dispose();
            Tensor keep_mask;
            // 原Python：if not self.use_m2m:
            if (!use_m2m)
            {
                // 原Python：if self.pred_iou_thresh > 0.0:
                if (pred_iou_thresh > 0.0f)
                {
                    keep_mask = data.iou_preds > pred_iou_thresh;
                    data.Filter(keep_mask.nonzero().squeeze(1));
                    keep_mask.Dispose();
                }

                // 原Python：data["stability_score"] = calculate_stability_score(...)
                data.stability_score = calculate_stability_score(
                    data.masks,
                    mask_threshold,
                    stability_score_offset
                );

                // 原Python：if self.stability_score_thresh > 0.0:
                if (stability_score_thresh > 0.0f)
                {
                    keep_mask = data.stability_score >= stability_score_thresh;
                    data.Filter(keep_mask.nonzero().squeeze(1));
                    keep_mask.Dispose();
                }
            }
            // 原Python：else:
            else
            {
                // 原Python：in_points = self.predictor._transforms.transform_coords(...)
                in_points = predictor.transforms.transform_coords(data.points, normalize: normalize,orig_hw: (im_size.Item1, im_size.Item2));

                // 原Python：labels = torch.ones(...)
                Tensor labels = torch.ones(
                    in_points.shape[0],
                    dtype: ScalarType.Int32,
                    device: in_points.device
                );

                // 原Python：masks, ious = self.refine_with_m2m(...)
                var (refined_masks, ious) = refine_with_m2m(
                    in_points,
                    labels,
                    data.low_res_masks,
                    points_per_batch
                );

                // 原Python：data["masks"] = masks.squeeze(1)
                data.masks = refined_masks.squeeze(1);
                // 原Python：data["iou_preds"] = ious.squeeze(1)
                data.iou_preds = ious.squeeze(1);

                // 原Python：if self.pred_iou_thresh > 0.0:
                if (pred_iou_thresh > 0.0f)
                {
                    keep_mask = data.iou_preds > pred_iou_thresh;
                    data.Filter(keep_mask.nonzero().squeeze(1));
                    keep_mask.Dispose();
                }

                // 原Python：data["stability_score"] = calculate_stability_score(...)
                data.stability_score = calculate_stability_score(
                    data.masks,
                    mask_threshold,
                    stability_score_offset
                );

                // 原Python：if self.stability_score_thresh > 0.0:
                if (stability_score_thresh > 0.0f)
                {
                    keep_mask = data.stability_score >= stability_score_thresh;
                    data.Filter(keep_mask.nonzero().squeeze(1));
                    keep_mask.Dispose();
                }
            }

            // 原Python：data["masks"] = data["masks"] > self.mask_threshold
            data.masks = data.masks > mask_threshold;

            // 原Python：data["boxes"] = batched_mask_to_box(data["masks"])
            data.boxes = batched_mask_to_box(data.masks);

            // 原Python：keep_mask = ~is_box_near_crop_edge(...)
            Tensor keep_mask1 = is_box_near_crop_edge(
                data.boxes,
                crop_box.ToArray(),
                new[] { orig_w, orig_h }
            );

            // 原Python：if not torch.all(keep_mask):
            if (!torch.all(keep_mask1).item<bool>())
            {
                data.Filter(keep_mask1.nonzero().squeeze(1));
            }
            keep_mask1.Dispose();

            // 原Python：data["masks"] = uncrop_masks(...)
            data.masks = uncrop_masks(
                data.masks,
                crop_box.ToArray(),
                orig_h,
                orig_w
            );

            // 原Python：data["rles"] = mask_to_rle_pytorch(data["masks"])
            data.Rles = mask_to_rle_pytorch(data.masks);

            // 原Python：del data["masks"]
            data.masks.Dispose();
            data.masks = null;

            // 原Python：return data
            return data;
        }

        // 以下辅助方法保持占位实现（略）
        //private List<Tensor> build_all_layer_point_grids(int pointsPerSide, int cropLayers, int downscaleFactor) => new List<Tensor>();
        //private (List<List<int>>, List<int>) generate_crop_boxes((long, long) origSize, int cropLayers, float overlapRatio) => (new List<List<int>>(), new List<int>());
        //private Tensor batched_nms(Tensor boxes, Tensor scores, Tensor categories, float iouThreshold) => torch.tensor(new int[0]);
        //private List<Tensor> batch_iterator(int batchSize, Tensor points) => new List<Tensor> { points };
        //private Tensor uncrop_boxes_xyxy(Tensor box, List<int> cropBox) => box;
        //private Tensor uncrop_points(Tensor point, List<int> cropBox) => point;
        //private List<Tensor> calculate_stability_score(List<Tensor> masks, float mask_threshold, float offset) => new List<Tensor>();
        //private (Tensor, Tensor) refine_with_m2m(Tensor points, Tensor labels, Tensor lowResmasks, int batchSize) => (torch.zeros(0), torch.zeros(0));
        //private List<Tensor> batched_mask_to_box(List<Tensor> masks) => new List<Tensor>();
        //private Tensor is_box_near_crop_edge(Tensor boxes, List<int> cropBox, int[] origSize) => torch.zeros(boxes.shape[0], dtype: ScalarType.Bool);
        //private Tensor uncrop_masks(Tensor mask, List<int> cropBox, int origH, int origW) => mask;
        //private List<Rle> mask_to_rle_pytorch(List<Tensor> masks) => new List<Rle>();

        //private static Sam2Base BuildSam2Hf(string modelId, Dictionary<string, object> kwargs) => throw new NotImplementedException();
        //private static object coco_encode_rle(Rle rle) => throw new NotImplementedException();
        //private static Tensor rle_to_mask(Rle rle) => throw new NotImplementedException();
        //private static float area_from_rle(Rle rle) => throw new NotImplementedException();
        //private static Tensor box_xyxy_to_xywh(Tensor box) => box;
        //private static Tensor box_area(List<Tensor> boxes) => torch.zeros(boxes.Count);
        /// <summary>
        /// 原Python：@staticmethod def postprocess_small_regions（完全保留命名/逻辑）
        /// </summary>
        /// <param name="mask_data">原Python：mask_data: MaskData</param>
        /// <param name="min_area">原Python：min_area: int</param>
        /// <param name="nms_thresh">原Python：nms_thresh: float</param>
        /// <returns>原Python：-> MaskData</returns>
        public static MaskData postprocess_small_regions(MaskData mask_data,int min_area,float nms_thresh)
        {
            // 原Python：if len(mask_data["rles"]) == 0: return mask_data
            if (mask_data.Rles == null || mask_data.Rles.Count == 0)
            {
                return mask_data;
            }

            // 原Python：new_masks = []; scores = []
            List<Tensor> new_masks = new List<Tensor>();
            List<float> scores = new List<float>();

            // 原Python：for rle in mask_data["rles"]:
            foreach (Rle rle in mask_data.Rles)
            {
                // 原Python：mask = rle_to_mask(rle)
                Tensor mask =rle_to_mask(rle);

                // 原Python：mask, changed = remove_small_regions(mask, min_area, mode="holes")
                var (maskAfterHoles, changedHoles) = remove_small_regions(mask, min_area, "holes");
                bool unchanged = !changedHoles;

                // 原Python：mask, changed = remove_small_regions(mask, min_area, mode="islands")
                var (maskAfterIslands, changedIslands) = remove_small_regions(maskAfterHoles, min_area, "islands");
                unchanged = unchanged && !changedIslands;

                // 原Python：new_masks.append(torch.as_tensor(mask).unsqueeze(0))
                Tensor maskTensor = maskAfterIslands;
                new_masks.Add(maskTensor);

                // 原Python：scores.append(float(unchanged))
                scores.Add(unchanged ? 1.0f : 0.0f);

                // 释放OpenCV Mat资源
                mask.Dispose();
                maskAfterHoles.Dispose();
                maskAfterIslands.Dispose();
            }

            // 原Python：masks = torch.cat(new_masks, dim=0)
            Tensor masks = torch.cat(new_masks.ToArray(), dim: 0);

            // 原Python：boxes = batched_mask_to_box(masks)
            Tensor boxes = batched_mask_to_box(masks);

            // 原Python：keep_by_nms = batched_nms(...)
            Tensor keep_by_nms = batched_nms(
                boxes.to(ScalarType.Float32),
                torch.tensor(scores.ToArray(), dtype: ScalarType.Float32),
                torch.zeros_like(boxes[.., 0], dtype: ScalarType.Float32), // categories
                iou_threshold: nms_thresh
            );

            // 原Python：for i_mask in keep_by_nms:
            long[] keepIndices = keep_by_nms.data<long>().ToArray<long>();
            foreach (long i_mask in keepIndices)
            {
                int idx = (int)i_mask;
                // 原Python：if scores[i_mask] == 0.0:
                if (Math.Abs(scores[idx] - 0.0f) < 1e-6f)
                {
                    // 原Python：mask_torch = masks[i_mask].unsqueeze(0)
                    Tensor mask_torch = masks[idx].unsqueeze(0);

                    // 原Python：mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                    mask_data.Rles[idx] = mask_to_rle_pytorch(mask_torch)[0];

                    // 原Python：mask_data["boxes"][i_mask] = boxes[i_mask]
                    mask_data.boxes[idx] = boxes[idx]; // 直接更新box值

                    mask_torch.Dispose();
                }
            }

            // 原Python：mask_data.filter(keep_by_nms)
            mask_data.Filter(keep_by_nms);

            // 释放临时Tensor资源
            masks.Dispose();
            boxes.Dispose();
            keep_by_nms.Dispose();
            new_masks.ForEach(t => t.Dispose());

            // 原Python：return mask_data
            return mask_data;
        }

        private (Tensor, Tensor) refine_with_m2m(Tensor in_points, Tensor labels, Tensor low_res_masks, int points_per_batch)
        {
            // 原Python逻辑：M2M细化（此处为示例实现，需替换为你的实际逻辑）
            Tensor masks = low_res_masks.unsqueeze(1);
            Tensor ious = torch.ones(low_res_masks.shape[0], 1, dtype: ScalarType.Float32, device: low_res_masks.device);
            return (masks, ious);
        }
    }

    // Remove duplicate SAM2ImagePredictor and Transform definitions from this file to avoid conflicts.

    // Helper to convert tensor arrays when unbind returns array
    internal static class TensorHelpers
    {
        public static List<Tensor> ToList(Tensor[] arr)
        {
            if (arr == null) return new List<Tensor>();
            return new List<Tensor>(arr);
        }
    }
}