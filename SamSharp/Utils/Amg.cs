using OpenCvSharp;
using System.Collections;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static OpenCvSharp.FileStorage;
using static Sam2Sharp.Utils.Classes;
using static TorchSharp.torch;

namespace Sam2Sharp.Utils
{
    //public class MaskData : IDisposable
    //{
    //    public List<Tensor> iou_preds { get; set; }

    //    public List<Tensor> points { get; set; }

    //    public List<Tensor> stability_score { get; set; }
    //    public List<Tensor> boxes { get; set; }
    //    public List<Tensor> crop_boxes { get; set; }

    //    public List<Tensor> masks { get; set; }
    //    public List<Tensor> low_res_masks { get; set; }       

    //    public List<Rle> Rles { get; set; }

    //    public void Cat(MaskData data)
    //    {
    //        //iou_preds = iou_preds is null ? data.iou_preds : torch.concat(new List<Tensor>[] { iou_preds, data.iou_preds });
    //        //points = points is null ? data.points : torch.concat(new List<Tensor>[] { points, data.points });
    //        //stability_score = stability_score is null ? data.stability_score : torch.concat(new List<Tensor>[] { stability_score, data.stability_score });
    //        //boxes = boxes is null ? data.boxes : torch.concat(new List<Tensor>[] { boxes, data.boxes });
    //        //crop_boxes = crop_boxes is null ? data.crop_boxes : torch.concat(new List<Tensor>[] { crop_boxes, data.crop_boxes });
    //        //masks = masks is null ? data.masks : torch.concat(new List<Tensor>[] { masks, data.masks });
    //        if(iou_preds is null&& data.iou_preds is null)
    //        {
    //            iou_preds.AddRange(data.iou_preds);
    //        }
    //        else
    //        {
    //            iou_preds=iou_preds ?? data.iou_preds;
    //        }
    //        if (points is null && data.points is null)
    //        {
    //            points.AddRange(data.points);
    //        }
    //        else
    //        {
    //            points = points ?? data.points;
    //        }
    //        if (stability_score is null && data.stability_score is null)
    //        {
    //            stability_score.AddRange(data.stability_score);
    //        }
    //        else
    //        {
    //            stability_score = stability_score ?? data.stability_score;
    //        }
    //        if (boxes is null && data.boxes is null)
    //        {
    //            boxes.AddRange(data.boxes);
    //        }
    //        else
    //        {
    //            boxes = boxes ?? data.boxes;
    //        }
    //        if (crop_boxes is null && data.crop_boxes is null)
    //        {
    //            crop_boxes.AddRange(data.crop_boxes);
    //        }
    //        else
    //        {
    //            crop_boxes = crop_boxes ?? data.crop_boxes;
    //        }
    //        if (masks is null && data.masks is null)
    //        {
    //            masks.AddRange(data.masks);
    //        }
    //        else
    //        {
    //            masks = masks ?? data.masks;
    //        }
    //        //iou_preds = iou_preds is null ? data.iou_preds : data.iou_preds is null ? iou_preds : iou_preds.AddRange(data.iou_preds); };
    //        //points = points is null ? data.points : torch.concat(new List<Tensor>[] { points, data.points });
    //        //stability_score = stability_score is null ? data.stability_score : torch.concat(new List<Tensor>[] { stability_score, data.stability_score });
    //        //boxes = boxes is null ? data.boxes : torch.concat(new List<Tensor>[] { boxes, data.boxes });
    //        //crop_boxes = crop_boxes is null ? data.crop_boxes : torch.concat(new List<Tensor>[] { crop_boxes, data.crop_boxes });
    //        //masks = masks is null ? data.masks : torch.concat(new List<Tensor>[] { masks, data.masks });
    //        if (Rles is null)
    //        {
    //            Rles = data.Rles;
    //        }
    //        else
    //        {
    //            Rles.AddRange(data.Rles);
    //        }
    //    }

    //    public void Dispose()
    //    {
    //        iou_preds?.Clear();
    //        points?.Clear();
    //        stability_score?.Clear();
    //        boxes?.Clear();
    //        crop_boxes?.Clear();
    //        masks?.Clear();
    //        Rles.Clear();
    //    }

    //    public void Filter(Tensor index)
    //    {
    //        long[] indexes = index.data<long>().ToArray();
    //        List<Rle> newRles = new List<Rle>();
    //        foreach (long item in indexes)
    //        {
    //            if (Rles is not null)
    //            {
    //                newRles.Add(Rles[(int)item]);
    //            }
    //        }
    //        Rles = newRles;

    //        List<Tensor> newiou_preds = new List<Tensor>();
    //        foreach (long item in indexes)
    //        {
    //            if (iou_preds is not null)
    //            {
    //                newiou_preds.Add(iou_preds[(int)item]);
    //            }
    //        }
    //        iou_preds = newiou_preds;
    //        List<Tensor> newpoints = new List<Tensor>();
    //        foreach (long item in indexes)
    //        {
    //            if (points is not null)
    //            {
    //                newpoints.Add(points[(int)item]);
    //            }
    //        }
    //        points = newpoints;
    //        List<Tensor> newstability_score = new List<Tensor>();
    //        foreach (long item in indexes)
    //        {
    //            if (stability_score is not null)
    //            {
    //                newstability_score.Add(stability_score[(int)item]);
    //            }
    //        }
    //        stability_score = newstability_score;
    //        List<Tensor> newboxes = new List<Tensor>();
    //        foreach (long item in indexes)
    //        {
    //            if (boxes is not null)
    //            {
    //                newboxes.Add(boxes[(int)item]);
    //            }
    //        }
    //        boxes = newboxes;
    //        List<Tensor> newcrop_boxes = new List<Tensor>();
    //        foreach (long item in indexes)
    //        {
    //            if (crop_boxes is not null)
    //            {
    //                newcrop_boxes.Add(crop_boxes[(int)item]);
    //            }
    //        }
    //        crop_boxes = newcrop_boxes;
    //        List<Tensor> newmasks = new List<Tensor>();
    //        foreach (long item in indexes)
    //        {
    //            if (masks is not null)
    //            {
    //                newmasks.Add(masks[(int)item]);
    //            }
    //        }
    //        masks = newmasks;

    //        //iou_preds = iou_preds[torch.as_tensor(index, device: iou_preds.device)];
    //        //points = points[torch.as_tensor(index, device: points.device)];
    //        //stability_score = stability_score[torch.as_tensor(index, device: stability_score.device)];
    //        //boxes = boxes[torch.as_tensor(index, device: boxes.device)];
    //        //crop_boxes = crop_boxes[torch.as_tensor(index, device: crop_boxes.device)];
    //        //if (masks is not null)
    //        //{
    //        //    masks = masks[torch.as_tensor(index, device: masks.device)];
    //        //}
    //    }
    //}

    public class MaskData : IDisposable
    {
        private bool disposedValue;

        // 核心属性：单Tensor（批量维度拼接），仅Rles为List<Rle>
        public Tensor iou_preds { get; set; }
        public Tensor points { get; set; }
        public Tensor stability_score { get; set; }
        public Tensor boxes { get; set; }
        public Tensor crop_boxes { get; set; }
        public Tensor masks { get; set; }
        public Tensor low_res_masks { get; set; }
        public List<Rle> Rles { get; set; }

        // 构造函数：初始化Rles，避免空引用
        public MaskData()
        {
            Rles = new List<Rle>(); // 必须初始化，防止AddRange空引用
                                    // Tensor属性初始化为null，按需赋值
            iou_preds = null;
            points = null;
            stability_score = null;
            boxes = null;
            crop_boxes = null;
            masks = null;
            low_res_masks = null;
        }

        #region 修复后的Cat方法（Tensor拼接 + Rles合并）
        public void Cat(MaskData data)
        {
            if (data == null) return;

            // 核心逻辑：Tensor拼接（dim=0，批量维度），兼容null值
            iou_preds = ConcatTensor(iou_preds, data.iou_preds);
            points = ConcatTensor(points, data.points);
            stability_score = ConcatTensor(stability_score, data.stability_score);
            boxes = ConcatTensor(boxes, data.boxes);
            crop_boxes = ConcatTensor(crop_boxes, data.crop_boxes);
            masks = ConcatTensor(masks, data.masks);
            low_res_masks = ConcatTensor(low_res_masks, data.low_res_masks);

            // 合并Rles（仅保留一次，修复重复添加问题）
            if (data.Rles != null && data.Rles.Count > 0)
            {
                Rles.AddRange(data.Rles);
            }
        }

        /// <summary>
        /// 辅助方法：安全拼接两个Tensor（处理null、维度匹配）
        /// </summary>
        private Tensor ConcatTensor(Tensor target, Tensor source)
        {
            // 1. 源Tensor为null，直接返回目标Tensor
            if ((source is null|| source.numel()==0) && target is not null)
                return target;

            // 2. 目标Tensor为null，返回源Tensor的克隆（避免原Tensor被外部修改）
            if ((target is null||target.numel()==0) && source is not null)
                return source.clone();

            // 3. 校验维度匹配（除批量维度dim=0外，其他维度必须一致）
            var targetShape = target.shape;
            var sourceShape = source.shape;
            if (targetShape.Length != sourceShape.Length)
                throw new ArgumentException($"Tensor维度不匹配：目标{targetShape.Length}维，源{sourceShape.Length}维");

            for (int i = 1; i < targetShape.Length; i++)
            {
                if (targetShape[i] != sourceShape[i])
                    throw new ArgumentException($"Tensor第{i}维尺寸不匹配：目标{targetShape[i]}，源{sourceShape[i]}");
            }

            // 4. 在批量维度（dim=0）拼接
            return torch.concat(new[] { target, source }, dim: 0);
        }
        #endregion

        #region 修复后的Filter方法（安全索引Tensor + Rles筛选）
        public void Filter(Tensor index)
        {
            if (index is null || index.numel() == 0)
            {
                // 空索引：清空所有数据
                ClearAllData();
                return;
            }

            // 1. 处理Rles筛选（兼容null）
            if (Rles != null && Rles.Count > 0)
            {
                long[] indexes = index.data<long>().ToArray();
                List<Rle> newRles = new List<Rle>();
                foreach (long item in indexes)
                {
                    int idx = (int)item;
                    if (idx >= 0 && idx < Rles.Count)
                        newRles.Add(Rles[idx]);
                }
                Rles = newRles;
            }

            // 2. 处理Tensor筛选（安全索引，兼容null）
            iou_preds = SafeIndexTensor(iou_preds, index);
            points = SafeIndexTensor(points, index);
            stability_score = SafeIndexTensor(stability_score, index);
            boxes = SafeIndexTensor(boxes, index);
            crop_boxes = SafeIndexTensor(crop_boxes, index);
            masks = SafeIndexTensor(masks, index);
            low_res_masks = SafeIndexTensor(low_res_masks, index);
        }

        /// <summary>
        /// 辅助方法：安全索引Tensor（处理null、设备一致性）
        /// </summary>
        private Tensor SafeIndexTensor(Tensor tensor, Tensor index)
        {
            if (tensor is null || tensor.numel() == 0)
                return null;

            // 确保索引与Tensor在同一设备（CPU/GPU）
            var indexOnDevice = index.to(device: tensor.device);
            try
            {
                return tensor[indexOnDevice];
            }
            finally
            {
                indexOnDevice.Dispose(); // 释放临时索引张量
            }
        }

        /// <summary>
        /// 清空所有数据（辅助方法）
        /// </summary>
        private void ClearAllData()
        {
            // 释放Tensor资源并置null
            iou_preds?.Dispose();
            points?.Dispose();
            stability_score?.Dispose();
            boxes?.Dispose();
            crop_boxes?.Dispose();
            masks?.Dispose();
            low_res_masks?.Dispose();

            iou_preds = null;
            points = null;
            stability_score = null;
            boxes = null;
            crop_boxes = null;
            masks = null;
            low_res_masks = null;

            // 清空Rles
            Rles?.Clear();
        }
        #endregion

        #region 完善的Dispose方法（释放所有Tensor资源）
        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // 释放托管资源：所有Tensor
                    iou_preds?.Dispose();
                    points?.Dispose();
                    stability_score?.Dispose();
                    boxes?.Dispose();
                    crop_boxes?.Dispose();
                    masks?.Dispose();
                    low_res_masks?.Dispose();

                    // 清空Rles
                    Rles?.Clear();
                }

                // 置空大型字段，帮助GC回收
                iou_preds = null;
                points = null;
                stability_score = null;
                boxes = null;
                crop_boxes = null;
                masks = null;
                low_res_masks = null;
                Rles = null;

                disposedValue = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        // 终结器：确保未托管资源释放（Tensor底层是C++资源）
        ~MaskData()
        {
            Dispose(disposing: false);
        }
        #endregion
    }

    public class Rle
    {
        public int[] Size { get; set; }
        public List<long> Counts { get; set; }

        // 构造函数：初始化属性，避免空引用
        public Rle()
        {
            Size = new int[0];
            Counts = new List<long>();
        }
    }

    #region 工具函数（方法名、变量名与Python完全一致）
    public static class AMG
    {
        public static Tensor is_box_near_crop_edge(Tensor boxes, List<int> crop_box, List<int> orig_box, float atol = 20.0f)
        {
            using var _ = NewDisposeScope();
            Tensor crop_box_torch = torch.as_tensor(crop_box, dtype: torch.float32, device: boxes.device);
            Tensor orig_box_torch = torch.as_tensor(orig_box, dtype: torch.float32, device: boxes.device);
            boxes = uncrop_boxes_xyxy(boxes, crop_box).@float();
            Tensor near_crop_edge = torch.isclose(boxes, crop_box_torch[TensorIndex.None, ..], atol: atol, rtol: 0);
            Tensor near_image_edge = torch.isclose(boxes, orig_box_torch[TensorIndex.None, ..], atol: atol, rtol: 0);
            near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge);
            return torch.any(near_crop_edge, dim: 1).MoveToOuterDisposeScope();
        }

        public static Tensor box_xyxy_to_xywh(Tensor box_xyxy)
        {
            var box_xywh = box_xyxy.clone();
            box_xywh[.., 2] = box_xywh[.., 2] - box_xywh[.., 0];
            box_xywh[.., 3] = box_xywh[.., 3] - box_xywh[.., 1];
            return box_xywh;
        }

        /// <summary>
        /// 按批次大小切分单个Tensor（批量维度dim=0）
        /// </summary>
        /// <param name="batch_size">批次大小（正整数）</param>
        /// <param name="tensor">输入Tensor（至少1维，dim=0为批量维度）</param>
        /// <returns>切分后的List<Tensor>，每个元素为单批次Tensor</returns>
        /// <exception cref="ArgumentException">输入不合法时抛出</exception>
        public static List<Tensor> batch_iterator(int batch_size, Tensor tensor)
        {
            // 1. 输入校验
            if (batch_size <= 0)
                throw new ArgumentOutOfRangeException(nameof(batch_size), "批次大小必须为正整数");
            if (tensor is null || tensor.numel() == 0)
                return new List<Tensor>();
            if (tensor.dim() < 1)
                throw new ArgumentException("输入Tensor至少需要1维（批量维度）", nameof(tensor));

            // 2. 获取批量维度总数量
            long totalCount = tensor.shape[0];
            List<Tensor> batches = new List<Tensor>();

            // 3. 按批次切分Tensor（dim=0）
            for (long b = 0; b < totalCount; b += batch_size)
            {
                long start = b;
                long end = Math.Min(b + batch_size, totalCount);

                // 切分当前批次：tensor[start:end, ...]
                Tensor batch = tensor[TensorIndex.Slice(start, end)];
                batches.Add(batch);
            }
            return batches;
        }
        //public static List<Dictionary<string, object>> mask_to_rle_pytorch(Tensor tensor)
        //{
        //    var (b, h, w) = (tensor.size(0), tensor.size(1), tensor.size(2));
        //    tensor = tensor.permute(0, 2, 1).flatten(1);

        //    var diff = tensor[.., 1..] ^ tensor[.., ..^1];
        //    var change_indices = diff.nonzero();

        //    var output = new List<Dictionary<string, object>>();
        //    for (int i = 0; i < b; i++)
        //    {
        //        var iLong = (long)i;
        //        var curIdxsMask = change_indices[.., 0] == iLong;
        //        var curIdxs = change_indices[curIdxsMask, 1];

        //        var start = torch.tensor(new[] { 0L }, dtype: curIdxs.dtype, device: curIdxs.device);
        //        var mid = curIdxs + 1;
        //        var end = torch.tensor(new[] { h * w }, dtype: curIdxs.dtype, device: curIdxs.device);
        //        curIdxs = torch.cat(new[] { start, mid, end }, dim: 0);

        //        var btw_idxs = curIdxs[1..] - curIdxs[..^1];
        //        var counts = new List<long>();
        //        if (tensor[i, 0].item<bool>()) counts.Add(0);
        //        counts.AddRange(btw_idxs.detach().cpu().data<long>().ToArray());

        //        output.Add(new Dictionary<string, object>
        //    {
        //        { "size", new List<long> { h, w } },
        //        { "counts", counts }
        //    });
        //    }

        //    return output;


        //    //using var _ = NewDisposeScope();
        //    //// Put in fortran order and flatten h,w
        //    //long b = tensor.shape[0];
        //    //long h = tensor.shape[1];
        //    //long w = tensor.shape[2];
        //    //tensor = tensor.permute(0, 2, 1).flatten(1);

        //    //// Compute change indices
        //    //Tensor diff = tensor[.., 1..] ^ tensor[.., ..(int)(tensor.shape[1] - 1)];
        //    //Tensor change_indices = diff.nonzero();

        //    //// Encode run length
        //    //List<Rle> @out = new List<Rle>();
        //    //for (int i = 0; i < b; i++)
        //    //{
        //    //    Tensor cur_idxs = change_indices[change_indices[.., 0] == i, 1];

        //    //    cur_idxs = torch.cat(new Tensor[] {

        //    //    torch.tensor(new long []{ 0 }, dtype : cur_idxs.dtype, device : cur_idxs.device),
        //    //    cur_idxs + 1,
        //    //    torch.tensor(new long []{ h * w }, dtype : cur_idxs.dtype, device : cur_idxs.device),
        //    //});

        //    //    Tensor btw_idxs = cur_idxs[1..] - cur_idxs[..(int)(cur_idxs.shape[0] - 1)];

        //    //    List<long> counts = (tensor[i, 0].ToSingle() == 0) ? new List<long> { 0 } : new List<long>();
        //    //    counts.AddRange(btw_idxs.data<long>().ToArray());


        //    //    @out.Add(new Rle { Size = new int[] { (int)h, (int)w }, Counts = counts });
        //    //}
        //    //return @out;
        //}
        public static List<Rle> mask_to_rle_pytorch(List<Tensor> ttensor)
        {
            // var tensor = torch.stack(ttensor.ToArray());
            var tensor = torch.concat(ttensor.ToArray());

            using var _ = NewDisposeScope();
            // Put in fortran order and flatten h,w
            long b = tensor.shape[0];
            long h = tensor.shape[1];
            long w = tensor.shape[2];
            tensor = tensor.permute(0, 2, 1).flatten(1);

            // Compute change indices
            Tensor diff = tensor[.., 1..] ^ tensor[.., ..(int)(tensor.shape[1] - 1)];
            Tensor change_indices = diff.nonzero();

            // Encode run length
            List<Rle> @out = new List<Rle>();
            for (int i = 0; i < b; i++)
            {
                Tensor cur_idxs = change_indices[change_indices[.., 0] == i, 1];

                cur_idxs = torch.cat(new Tensor[] {

                    torch.tensor(new long []{ 0 }, dtype : cur_idxs.dtype, device : cur_idxs.device),
                    cur_idxs + 1,
                    torch.tensor(new long []{ h * w }, dtype : cur_idxs.dtype, device : cur_idxs.device),
                });

                Tensor btw_idxs = cur_idxs[1..] - cur_idxs[..(int)(cur_idxs.shape[0] - 1)];

                List<long> counts = (tensor[i, 0].ToSingle() == 0) ? new List<long> { 0 } : new List<long>();
                counts.AddRange(btw_idxs.data<long>().ToArray());


                @out.Add(new Rle { Size = new int[] { (int)h, (int)w }, Counts = counts });
            }
            return @out;
        }

        public static Tensor rle_to_mask_tensor(Rle rle)
        {
            var size = rle.Size;
            var (h, w) = ((long)size[0], size[1]);
            var counts = rle.Counts;

            var mask = torch.empty(h * w);
            long idx = 0;
            bool parity = false;

            foreach (var count in counts)
            {
                for (long j = 0; j < count; j++)
                {
                    mask[idx + j] = parity;
                }
                idx += count;
                parity = !parity;
            }

            mask = mask.reshape(w, h);
            return mask;
        }

        public static long area_from_rle(Rle rle)
        {
            var counts = (List<long>)rle.Counts;
            return counts.Where((_, i) => i % 2 == 1).Sum();
        }

        //public static Tensor calculate_stability_score(Tensor masks, float mask_threshold, float threshold_offset)
        //{
        //    var intersections = (masks > (mask_threshold + threshold_offset))
        //        .sum(-1, dtype: ScalarType.Int16)
        //        .sum(-1, dtype: ScalarType.Int32);

        //    var unions = (masks > (mask_threshold - threshold_offset))
        //        .sum(-1, dtype: ScalarType.Int16)
        //        .sum(-1, dtype: ScalarType.Int32);

        //    return intersections.to(ScalarType.Float) / unions.to(ScalarType.Float);
        //}

        //public static NDArray build_point_grid(int n_per_side)
        //{
        //    var offset = 1.0f / (2 * n_per_side);
        //    var points_one_side = np.linspace(offset, 1 - offset, n_per_side);

        //    var points_x = np.tile(points_one_side[np.newaxis, :], new[] { n_per_side, 1 });
        //    var points_y = np.tile(points_one_side[:, np.newaxis], new[] { 1, n_per_side });

        //    var points = np.stack(new[] { points_x, points_y }, axis: -1).reshape(-1, 2);
        //    return points;
        //}
        /// <summary>
        /// Generates a 2D grid of points evenly spaced in [0,1]x[0,1].
        /// </summary>
        /// <param name="n_per_side"></param>
        /// <returns></returns>
        internal static Tensor build_point_grid(int n_per_side)
        {
            using var _ = NewDisposeScope();
            float offset = 1 / (2 * n_per_side);
            Tensor points_one_side = torch.linspace(offset, 1 - offset, n_per_side);
            Tensor points_x = torch.tile(points_one_side[TensorIndex.None, ..], new long[] { n_per_side, 1 });
            Tensor points_y = torch.tile(points_one_side[.., TensorIndex.None], new long[] { 1, n_per_side });
            Tensor points = torch.stack(new Tensor[] { points_x, points_y }, dim: -1).reshape(-1, 2);
            return points.MoveToOuterDisposeScope();
        }
        /// <summary>
        ///  Computes the stability score for a batch of masks. The stability		score is the IoU between the binary masks obtained by thresholding		the predicted mask logits at high and low values.
        /// </summary>
        internal static Tensor calculate_stability_score(List<Tensor> tmasks, float mask_threshold, float threshold_offset)
        {
            using var _ = NewDisposeScope();
            // One mask is always contained inside the other.
            // Save memory by preventing unnecessary cast to torch.int64
            var masks = torch.concat(tmasks.ToArray());
            Tensor intersections = (
                    (masks > (mask_threshold + threshold_offset))
                    .sum(-1, type: torch.int16)
                    .sum(-1, type: torch.int32)
                );

            Tensor unions = (
                    (masks > (mask_threshold - threshold_offset))
                    .sum(-1, type: torch.int16)
                    .sum(-1, type: torch.int32)
                );

            return (intersections / unions).MoveToOuterDisposeScope();
        }

        public static List<Tensor> build_all_layer_point_grids(int n_per_side, int n_layers, int scale_per_layer)
        {
            var points_by_layer = new List<Tensor>();
            for (int i = 0; i <= n_layers; i++)
            {
                var n_points = (int)(n_per_side / Math.Pow(scale_per_layer, i));
                points_by_layer.Add(build_point_grid(n_points));
            }
            return points_by_layer;
        }

        public static (List<List<int>>, List<int>) generate_crop_boxes((int, int) im_size, int n_layers, float overlap_ratio)
        {
            var crop_boxes = new List<List<int>>();
            var layer_idxs = new List<int>();
            var (im_h, im_w) = (im_size.Item1, im_size.Item2);
            var short_side = Math.Min(im_h, im_w);

            // 原始图像
            crop_boxes.Add(new List<int> { 0, 0, im_w, im_h });
            layer_idxs.Add(0);

            Func<int, int, int, int> crop_len = (orig_len, n_crops, overlap) =>
            {
                return (int)Math.Ceiling((overlap * (n_crops - 1) + orig_len) / (double)n_crops);
            };

            for (int i_layer = 0; i_layer < n_layers; i_layer++)
            {
                var n_crops_per_side = (int)Math.Pow(2, i_layer + 1);
                var overlap = (int)(overlap_ratio * short_side * (2.0 / n_crops_per_side));

                var crop_w = crop_len(im_w, n_crops_per_side, overlap);
                var crop_h = crop_len(im_h, n_crops_per_side, overlap);

                var crop_box_x0 = new List<int>();
                var crop_box_y0 = new List<int>();
                for (int i = 0; i < n_crops_per_side; i++)
                {
                    crop_box_x0.Add((crop_w - overlap) * i);
                    crop_box_y0.Add((crop_h - overlap) * i);
                }

                // 笛卡尔积
                foreach (var x0 in crop_box_x0)
                {
                    foreach (var y0 in crop_box_y0)
                    {
                        var box = new List<int>
                        {
                            x0,
                            y0,
                            Math.Min(x0 + crop_w, im_w),
                            Math.Min(y0 + crop_h, im_h)
                        };
                        crop_boxes.Add(box);
                        layer_idxs.Add(i_layer + 1);
                    }
                }
            }

            return (crop_boxes, layer_idxs);
        }

        public static Tensor uncrop_boxes_xyxy(Tensor boxes, List<int> crop_box)
        {
            var x0 = crop_box[0];
            var y0 = crop_box[1];
            var offset = torch.tensor(new[,] { { x0, y0, x0, y0 } }, device: boxes.device);

            if (boxes.ndim == 3)
            {
                offset = offset.unsqueeze(1);
            }

            return boxes + offset;
        }

        public static Tensor uncrop_points(Tensor points, List<int> crop_box)
        {
            var x0 = crop_box[0];
            var y0 = crop_box[1];
            var offset = torch.tensor(new[,] { { x0, y0 } }, device: points.device);

            if (points.ndim == 3)
            {
                offset = offset.unsqueeze(1);
            }

            return points + offset;
        }

        public static Tensor uncrop_masks(Tensor masks, List<int> crop_box, int orig_h, int orig_w)
        {
            var x0 = crop_box[0];
            var y0 = crop_box[1];
            var x1 = crop_box[2];
            var y1 = crop_box[3];

            if (x0 == 0 && y0 == 0 && x1 == orig_w && y1 == orig_h)
            {
                return masks;
            }

            var pad_x = orig_w - (x1 - x0);
            var pad_y = orig_h - (y1 - y0);
            var pad = new long[] { x0, pad_x - x0, y0, pad_y - y0 };

            return torch.nn.functional.pad(masks, pad, value: 0);
        }


        //// 注：coco_encode_rle依赖pycocotools，C#无直接对应库，此处省略（可通过自定义实现COCO RLE编码）
        //public static Rle coco_encode_rle(Rle uncompressed_rle)
        //{
        //    var h = uncompressed_rle.Size[0];
        //    var w = uncompressed_rle.Size[1];
        //    var rle = new Rle();
        //    rle.Size = [h, w];
        //    rle.Counts = rle.Counts.decode("utf-8");  // Necessary to serialize with json
        //    return rle;
        //}
        /// <summary>
        /// RLE编码转换为掩码Mat（对应Python rle_to_mask，适配自定义Rle类）
        /// </summary>
        public static Mat rle_to_mask_mat(Rle rle)
        {
            if (rle == null || rle.Size == null || rle.Counts == null)
                throw new ArgumentNullException("Invalid Rle data, cannot convert to mask.");

            int height = rle.Size[0];
            int width = rle.Size[1];
            Mat mask = new Mat(height, width, MatType.CV_8UC1);

            int idx = 0;
            bool isForeground = false;

            foreach (var count in rle.Counts)
            {
                for (long i = 0; i < count; i++)
                {
                    if (idx >= height * width) break;
                    int y = idx / width;
                    int x = idx % width;
                    mask.At<byte>(y, x) = isForeground ? (byte)255 : (byte)0;
                    idx++;
                }
                isForeground = !isForeground;
            }

            return mask;
        }

        ///// <summary>
        ///// 掩码Tensor转换为RLE编码（对应Python mask_to_rle_pytorch，适配自定义Rle类）
        ///// </summary>
        //public static Rle mask_to_rle_pytorch(Tensor maskTensor)
        //{
        //    if (maskTensor is null) throw new ArgumentNullException(nameof(maskTensor));

        //    // 去除批次维度，转换为Mat
        //    var maskSqueezed = maskTensor.squeeze(0);
        //    var maskArray = maskSqueezed.ToArray<byte>();
        //    int height = (int)maskSqueezed.shape[0];
        //    int width = (int)maskSqueezed.shape[1];
        //    Mat mask = new Mat(height, width, MatType.CV_8UC1, maskArray);

        //    // 生成RLE编码
        //    List<long> counts = new List<long>();
        //    long currentCount = 0;
        //    bool currentIsForeground = mask.At<byte>(0, 0) == 255;

        //    for (int y = 0; y < height; y++)
        //    {
        //        for (int x = 0; x < width; x++)
        //        {
        //            bool isForeground = mask.At<byte>(y, x) == 255;
        //            if (isForeground == currentIsForeground)
        //            {
        //                currentCount++;
        //            }
        //            else
        //            {
        //                counts.Add(currentCount);
        //                currentCount = 1;
        //                currentIsForeground = isForeground;
        //            }
        //        }
        //    }
        //    counts.Add(currentCount);

        //    // 封装为自定义Rle类
        //    return new Rle
        //    {
        //        Size = new int[] { height, width },
        //        Counts = counts
        //    };
        //}

        ///// <summary>
        ///// 批量掩码转换为边框（对应Python batched_mask_to_box，输入List<Tensor>）
        ///// </summary>
        //public static List<Tensor> batched_mask_to_box(List<Tensor> masks)
        //{
        //    if (masks == null || masks.Count == 0) return new List<Tensor>();

        //    List<Tensor> boxes = new List<Tensor>();
        //    foreach (var mask in masks)
        //    {
        //        Tensor box = torch.zeros(4, dtype: ScalarType.Float32);
        //        var nonZero = mask.nonzero();

        //        if (nonZero.nelement() == 0)
        //        {
        //            boxes.Add(box);
        //            continue;
        //        }

        //        var yCoords = nonZero[.., 0].to(ScalarType.Float32);
        //        var xCoords = nonZero[.., 1].to(ScalarType.Float32);
        //        float x1 = xCoords.min().item<float>();
        //        float y1 = yCoords.min().item<float>();
        //        float x2 = xCoords.max().item<float>();
        //        float y2 = yCoords.max().item<float>();

        //        box = torch.tensor(new[] { x1, y1, x2, y2 }, dtype: ScalarType.Float32);
        //        boxes.Add(box);
        //    }

        //    return boxes;
        //}

        /// <summary>
        /// 批量NMS（对应Python batched_nms，适配List<Tensor>输入）
        /// </summary>
        public static Tensor batched_nms(List<Tensor> boxes, List<float> scores, float iouThreshold)
        {
            if (boxes == null || boxes.Count == 0) return torch.tensor(new long[0], dtype: ScalarType.Float32);

            // 转换为批量Tensor进行NMS计算
            Tensor boxesTensor = torch.stack(boxes.ToArray(), dim: 0);
            Tensor scoresTensor = torch.tensor(scores.ToArray(), dtype: ScalarType.Float32);
            Tensor categoriesTensor = torch.zeros_like(boxesTensor[.., 0], dtype: ScalarType.Float32);

            return torchvision.ops.nms(boxesTensor, scoresTensor, iouThreshold);
        }

        /// <summary>
        /// 批量迭代器
        /// </summary>
        public static IEnumerable<List<List<Tensor>>> batch_iterator(int batchSize, params List<Tensor>[] tensorLists)
        {
            if (tensorLists.Length == 0 || tensorLists[0] == null || tensorLists[0].Count == 0)
                yield break;

            int totalCount = tensorLists[0].Count;
            for (int start = 0; start < totalCount; start += batchSize)
            {
                int end = Math.Min(start + batchSize, totalCount);
                List<List<Tensor>> batch = new List<List<Tensor>>();

                foreach (var tensorList in tensorLists)
                {
                    if (tensorList == null)
                    {
                        batch.Add(null);
                        continue;
                    }
                    List<Tensor> batchSegment = tensorList.GetRange(start, end - start);
                    batch.Add(batchSegment);
                }

                yield return batch;
            }
        }
        ///// <summary>
        ///// 移除掩码Tensor中的小区域（岛屿/孔洞）
        ///// </summary>
        ///// <param name="mask">输入二值化Tensor（元素为0/1，推荐UInt8类型）</param>
        ///// <param name="area_thresh">小区域面积阈值</param>
        ///// <param name="mode">处理模式：holes（孔洞）/islands（岛屿）</param>
        ///// <returns>处理后的掩码Tensor + 是否执行了移除操作</returns>
        //public unsafe static (Mat, bool) remove_small_regions(Tensor mask, float area_thresh, string mode)
        //{
        //    // 1. 验证模式参数合法性
        //    if (mode != "holes" && mode != "islands")
        //    {
        //        throw new ArgumentException("mode must be either 'holes' or 'islands'");
        //    }

        //    // 2. 初始化工作掩码：先将Tensor转换为可操作的数组，执行异或操作区分孔洞和岛屿
        //    var correct_holes = mode == "holes";
        //    var maskArray = mask.data<bool>().ToArray<bool>(); // 提取Tensor中的布尔值数组
        //    var workingMaskArray = new byte[maskArray.Length];
        //    for (int i = 0; i < maskArray.Length; i++)
        //    {
        //        // 异或操作：correct_holes ^ mask值（转换为UInt8）
        //        workingMaskArray[i] = (byte)((correct_holes ? 1 : 0) ^ (maskArray[i] ? 1 : 0));
        //    }

        //    // 3. 转换为OpenCV的Mat并自动释放资源（using语句）
        //    var rows = (int)mask.shape[0];
        //    var cols = (int)mask.shape[1];
        //    var src = new Mat(cols, rows, MatType.CV_8UC1);
        //    int length = rows * cols;
        //    byte[] mask1 = new byte[length];

        //    Buffer.BlockCopy(maskArray, 0, mask1, 0, length);
        //    Marshal.Copy(mask1, 0, src.Data, length);
        //    Mat regionsMat = new Mat();
        //    Mat stats = new Mat();
        //    Mat centroids = new Mat();
        //    // 4. 计算连通域及其统计信息（OpenCV核心功能不变）
        //    int n_labels = Cv2.ConnectedComponentsWithStats(src, regionsMat, stats, centroids, PixelConnectivity.Connectivity8);
        //    // 5. 提取所有连通域的面积（跳过背景标签0）
        //    var sizes = new List<int>();
        //    for (int i = 1; i < n_labels; i++)
        //    {
        //        sizes.Add(stats.At<int>(i, 4));
        //    }

        //    // 6. 筛选出面积小于阈值的小区域标签
        //    var small_regions = new List<int>();
        //    for (int i = 0; i < sizes.Count; i++)
        //    {
        //        if (sizes[i] < area_thresh)
        //        {
        //            small_regions.Add(i + 1); // 对应连通域标签（从1开始）
        //        }
        //    }

        //    // 7. 无小区域时直接返回原掩码
        //    if (small_regions.Count == 0)
        //    {
        //        regionsMat?.Dispose(); // 释放OpenCV返回的regions矩阵
        //        return (regionsMat, false);
        //    }

        //    // 8. 构建需要保留/填充的标签集合
        //    List<int> fill_labels;
        //    if (correct_holes)
        //    {
        //        // 孔洞模式：保留背景（0）和小区域（需要填充的孔洞）
        //        fill_labels = new List<int> { 0 };
        //        fill_labels.AddRange(small_regions);
        //    }
        //    else
        //    {
        //        // 岛屿模式：保留非小区域的标签，若为空则保留最大区域
        //        fill_labels = new List<int>();
        //        for (int i = 0; i < n_labels; i++)
        //        {
        //            if (!small_regions.Contains(i))
        //            {
        //                fill_labels.Add(i);
        //            }
        //        }

        //        // 边界处理：无保留区域时保留最大连通域
        //        if (fill_labels.Count == 0)
        //        {
        //            var maxIdx = sizes.IndexOf(sizes.Max());
        //            fill_labels.Add(maxIdx + 1);
        //        }
        //    }

        //    // 9. 转换连通域标签为数组，并调整为与原掩码一致的形状
        //    //var regionsData = new int[length];
        //    //Marshal.Copy(regionsMat.Data, regionsData, 0, length);
        //    var regions = new Mat(cols, rows, MatType.CV_8UC1);
        //    unsafe
        //    {
        //        uint* mat = (uint*)(regionsMat.Data);
        //        byte* dst = (byte*)(regions.Data);
        //        for (int i = 0; i < regionsMat.Data; i++)
        //        {
        //            if (fill_labels.Contains((int)*mat++))
        //                *dst++ = 1;
        //        }
        //    }

        //    // 12. 释放资源并返回结果
        //    //regions?.Dispose();
        //    return (regions, true);
        //}
        /// <summary>
        /// 原Python：def remove_small_regions(mask: Tensor, area_thresh: float, mode: str) -> Tuple[Tensor, bool]
        /// 核心逻辑：移除掩码中的小连通区域/孔洞，返回处理后的掩码和是否修改的标识
        /// </summary>
        /// <param name="mask">原Python：mask: Tensor → TorchSharp Tensor（shape [H,W]，dtype bool）</param>
        /// <param name="area_thresh">原Python：area_thresh: float → 面积阈值</param>
        /// <param name="mode">原Python：mode: str → "holes"（移除孔洞）/"islands"（移除小岛屿）</param>
        /// <returns>原Python：Tuple[Tensor, bool] → (处理后的掩码Tensor, 是否修改)</returns>
        public static (Tensor, bool) remove_small_regions(Tensor mask, float area_thresh, string mode)
        {
            // 原Python：assert mode in ["holes", "islands"]
            if (mode != "holes" && mode != "islands")
                throw new ArgumentException($"mode must be 'holes' or 'islands', got {mode}");

            // 原Python：correct_holes = mode == "holes"
            bool correct_holes = mode == "holes";

            var maskArray = mask.data<bool>().ToArray<bool>(); // 提取Tensor中的布尔值数组
            var workingMaskArray = new byte[maskArray.Length];
            for (int i = 0; i < maskArray.Length; i++)
            {
                // 异或操作：correct_holes ^ mask值（转换为UInt8）
                workingMaskArray[i] = (byte)((correct_holes ? 1 : 0) ^ (maskArray[i] ? 1 : 0));
            }

            // 3. 转换为OpenCV的Mat并自动释放资源（using语句）
            var rows = (int)mask.shape[0];
            var cols = (int)mask.shape[1];
            var src = new Mat(cols, rows, MatType.CV_8UC1);
            int length = rows * cols;
            byte[] mask1 = new byte[length];

            Buffer.BlockCopy(workingMaskArray, 0, mask1, 0, length);
            Marshal.Copy(mask1, 0, src.Data, length);
            Mat regionsMat = new Mat();
            Mat stats = new Mat();
            Mat centroids = new Mat();
            // 4. 计算连通域及其统计信息（OpenCV核心功能不变）
            int n_labels = Cv2.ConnectedComponentsWithStats(src, regionsMat, stats, centroids, PixelConnectivity.Connectivity8);
            // 5. 提取所有连通域的面积（跳过背景标签0）
            var sizes = new List<int>();
            for (int i = 1; i < n_labels; i++)
            {
                sizes.Add(stats.At<int>(i, 4));
            }

            // 6. 筛选出面积小于阈值的小区域标签
            var small_regions = new List<int>();
            for (int i = 0; i < sizes.Count; i++)
            {
                if (sizes[i] < area_thresh)
                {
                    small_regions.Add(i + 1); // 对应连通域标签（从1开始）
                }
            }

            // 7. 无小区域时直接返回原掩码
            if (small_regions.Count == 0)
            {
                regionsMat?.Dispose(); // 释放OpenCV返回的regions矩阵
                return (mask, false);
            }

            // 8. 构建需要保留/填充的标签集合
            List<int> fill_labels;
            if (correct_holes)
            {
                // 孔洞模式：保留背景（0）和小区域（需要填充的孔洞）
                fill_labels = new List<int> { 0 };
                fill_labels.AddRange(small_regions);
            }
            else
            {
                // 岛屿模式：保留非小区域的标签，若为空则保留最大区域
                fill_labels = new List<int>();
                for (int i = 0; i < n_labels; i++)
                {
                    if (!small_regions.Contains(i))
                    {
                        fill_labels.Add(i);
                    }
                }

                // 边界处理：无保留区域时保留最大连通域
                if (fill_labels.Count == 0)
                {
                    var maxIdx = sizes.IndexOf(sizes.Max());
                    fill_labels.Add(maxIdx + 1);
                }
            }

            // 9. 转换连通域标签为数组，并调整为与原掩码一致的形状
            //var regionsData = new int[length];
            //Marshal.Copy(regionsMat.Data, regionsData, 0, length);
            //var regions = new Mat(cols, rows, MatType.CV_8UC1);
            //unsafe
            //{
            //    uint* mat = (uint*)(regionsMat.Data);
            //    byte* dst = (byte*)(regions.Data);
            //    for (int i = 0; i < length; i++)
            //    {
            //        if (fill_labels.Contains((int)*mat++))
            //            *dst++ = 1;
            //    }
            //}


            bool[,] maskArray2 = new bool[(int)mask.shape[0], (int)mask.shape[1]];
            int[] regionsArray = new int[(int)mask.shape[0]*(int)mask.shape[1]];
            Marshal.Copy(regionsMat.Data, regionsArray, 0, length);
            //int idx = 0;
            unsafe
            {
                fixed (int* ptr = regionsArray)
                fixed (bool* maskArrayPtr = maskArray2)
                {
                    int* mat = ptr;
                    bool* dst = (maskArrayPtr);

                    for (int i = 0; i < length; i++)
                    {
                            int label = *mat++;
                            *dst++ = fill_labels.Contains(label);
                    }
                }
            }
            // 转换为TorchSharp Tensor
            Tensor result_mask = torch.tensor(maskArray2, dtype: ScalarType.Bool, device: mask.device);

            // 释放资源
            workingMaskArray = null;
            maskArray = null;
            src.Dispose();
            mask.Dispose();
            regionsMat.Dispose();
            regionsArray = null;

            stats.Dispose();
            maskArray2 = null;
            return (result_mask, true);
        }
        public static Tensor batched_mask_to_box(Tensor masks)
        {
            if (torch.numel(masks) == 0)
            {
                var mshape = masks.shape.Take((int)masks.ndim - 2).Concat(new[] { 4L }).ToArray();
                return torch.zeros(mshape, device: masks.device);
            }

            var shape = masks.shape;
            var h = shape[^2];
            var w = shape[^1];
            Tensor masksFlattened;

            if (masks.ndim > 2)
            {
                masksFlattened = masks.flatten(0, -3);
            }
            else
            {
                masksFlattened = masks.unsqueeze(0);
            }

            // 计算上下边缘
            var in_height = masksFlattened.max(dim: -1).values;
            var hRange = torch.arange(h, device: in_height.device);
            var in_height_coords = in_height * hRange.unsqueeze(0);

            var bottom_edges = in_height_coords.max(dim: -1).values;
            in_height_coords = in_height_coords + h * (~in_height);
            var top_edges = in_height_coords.min(dim: -1).values;

            // 计算左右边缘
            var in_width = masksFlattened.max(dim: -2).values;
            var wRange = torch.arange(w, device: in_width.device);
            var in_width_coords = in_width * wRange.unsqueeze(0);

            var right_edges = in_width_coords.max(dim: -1).values;
            in_width_coords = in_width_coords + w * (~in_width);
            var left_edges = in_width_coords.min(dim: -1).values;

            // 筛选空掩码
            var empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges);
            var outTensor = torch.stack(
                new[] { left_edges, top_edges, right_edges, bottom_edges }, dim: -1);
            outTensor = outTensor * (~empty_filter).unsqueeze(-1);

            // 还原原始形状
            if (masks.ndim > 2)
            {
                var newShape = shape.Take((int)masks.ndim - 2).Concat(new[] { 4L }).ToArray();
                outTensor = outTensor.reshape(newShape);
            }
            else
            {
                outTensor = outTensor[0];
            }

            return outTensor;
        }

        /// <summary>
        /// 原Python：def rle_to_mask(rle: Dict[str, Any]) -> Tensor
        /// 核心逻辑：从非压缩RLE计算二进制掩码（返回TorchSharp Tensor）
        /// </summary>
        /// <param name="rle">原Python：rle: Dict[str, Any] → 适配为自定义Rle类</param>
        /// <returns>原Python：Tensor → TorchSharp Tensor（shape [h, w]，dtype bool）</returns>
        public static Tensor rle_to_mask(Rle rle)
        {
            // 原Python：h, w = rle["size"]
            if (rle.Size == null || rle.Size.Length != 2)
                throw new ArgumentException("RLE size must be [h, w]");
            int h = rle.Size[0];
            int w = rle.Size[1];

            // 初始化一维掩码数组（对应原Python的np.empty(h*w, dtype=bool)）
            long totalPixels = (long)h * w;
            bool[] maskArray = new bool[totalPixels];
            long idx = 0;
            bool parity = false; // 初始为背景（False）

            // 原Python：for count in rle["counts"]:
            foreach (long count in rle.Counts)
            {
                if (idx >= totalPixels) break; // 超出范围则终止

                // 原Python：mask[idx : idx + count] = parity
                long end = Math.Min(idx + count, totalPixels);
                for (long i = idx; i < end; i++)
                {
                    maskArray[i] = parity;
                }

                // 原Python：idx += count; parity ^= True
                idx += count;
                parity = !parity; // 布尔异或，切换前景/背景
            }

            // 转换为TorchSharp Tensor并重塑维度（对应原Python reshape + transpose）
            // 1. 一维数组转Tensor：shape [h*w]
            Tensor maskTensor = torch.tensor(maskArray, dtype: ScalarType.Bool);
            // 2. 重塑为 [w, h] → 对应原Python mask.reshape(w, h)
            maskTensor = maskTensor.reshape(w, h);
            // 3. 转置为 [h, w] → 对应原Python mask.transpose()（C顺序）
            maskTensor = maskTensor.t();

            return maskTensor;
        }

        
        /// <summary>
        ///  Computes the stability score for a batch of masks. The stability		score is the IoU between the binary masks obtained by thresholding		the predicted mask logits at high and low values.
        /// </summary>
        internal static Tensor calculate_stability_score(Tensor masks, float mask_threshold, float threshold_offset)
        {
            using var _ = NewDisposeScope();
            // One mask is always contained inside the other.
            // Save memory by preventing unnecessary cast to torch.int64
            Tensor intersections = (
                    (masks > (mask_threshold + threshold_offset))
                    .sum(-1, type: torch.int16)
                    .sum(-1, type: torch.int32)
                );

            Tensor unions = (
                    (masks > (mask_threshold - threshold_offset))
                    .sum(-1, type: torch.int16)
                    .sum(-1, type: torch.int32)
                );

            return (intersections / unions).MoveToOuterDisposeScope();
        }

        /// <summary>
        /// Filter masks at the edge of a crop, but not at the edge of the original image.
        /// </summary>
        /// <param name="boxes"></param>
        /// <param name="crop_box"></param>
        /// <param name="ororig_box"></param>
        /// <param name="atol"></param>
        /// <returns></returns>
        internal static Tensor is_box_near_crop_edge(Tensor boxes, int[] crop_box, int[] orig_box, float atol = 20.0f)
        {
            using var _ = NewDisposeScope();
            Tensor crop_box_torch = torch.as_tensor(crop_box, dtype: torch.float32, device: boxes.device);
            Tensor orig_box_torch = torch.as_tensor(orig_box, dtype: torch.float32, device: boxes.device);
            boxes = uncrop_boxes_xyxy(boxes, crop_box).@float();
            Tensor near_crop_edge = torch.isclose(boxes, crop_box_torch[TensorIndex.None, ..], atol: atol, rtol: 0);
            Tensor near_image_edge = torch.isclose(boxes, orig_box_torch[TensorIndex.None, ..], atol: atol, rtol: 0);
            near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge);
            return torch.any(near_crop_edge, dim: 1).MoveToOuterDisposeScope();
        }

        internal static Tensor uncrop_boxes_xyxy(Tensor boxes, int[] crop_box)
        {
            int x0 = crop_box[0];
            int y0 = crop_box[1];
            Tensor offset = torch.tensor(new int[,] { { x0, y0, x0, y0 } }, device: boxes.device);
            // Check if boxes has a channel dimension
            if (boxes.shape.Length == 3)
            {
                offset = offset.unsqueeze(1);
            }

            return boxes + offset;
        }

        internal static Tensor uncrop_masks(Tensor masks, int[] crop_box, int orig_h, int orig_w)
        {
            int x0 = crop_box[0];
            int y0 = crop_box[1];
            int x1 = crop_box[2];
            int y1 = crop_box[3];

            if (x0 == 0 && y0 == 0 && x1 == orig_w && y1 == orig_h)
            {
                return masks;
            }

            // Coordinate transform masks
            int pad_x = orig_w - (x1 - x0);
            int pad_y = orig_h - (y1 - y0);
            long[] pad = new long[] { x0, pad_x - x0, y0, pad_y - y0 };
            return torch.nn.functional.pad(masks, pad, value: 0);
        }

        /// <summary>
        /// Encodes masks to an uncompressed RLE, in the format expected by	pycoco tools.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        internal static List<Rle> mask_to_rle_pytorch(Tensor tensor)
        {
            using var _ = NewDisposeScope();
            // Put in fortran order and flatten h,w
            long b = tensor.shape[0];
            long h = tensor.shape[1];
            long w = tensor.shape[2];
            tensor = tensor.permute(0, 2, 1).flatten(1);

            // Compute change indices
            Tensor diff = tensor[.., 1..] ^ tensor[.., ..(int)(tensor.shape[1] - 1)];
            Tensor change_indices = diff.nonzero();

            // Encode run length
            List<Rle> @out = new List<Rle>();
            for (int i = 0; i < b; i++)
            {
                Tensor cur_idxs = change_indices[change_indices[.., 0] == i, 1];

                cur_idxs = torch.cat(new Tensor[] {

                    torch.tensor(new long []{ 0 }, dtype : cur_idxs.dtype, device : cur_idxs.device),
                    cur_idxs + 1,
                    torch.tensor(new long []{ h * w }, dtype : cur_idxs.dtype, device : cur_idxs.device),
                });

                Tensor btw_idxs = cur_idxs[1..] - cur_idxs[..(int)(cur_idxs.shape[0] - 1)];

                List<long> counts = (tensor[i, 0].ToSingle() == 0) ? new List<long> { 0 } : new List<long>();
                counts.AddRange(btw_idxs.data<long>().ToArray());


                @out.Add(new Rle { Size = new int[] { (int)h, (int)w }, Counts = counts });
            }
            return @out;
        }

        internal static Tensor batched_nms(Tensor boxes, Tensor scores, Tensor idxs, float iou_threshold)
        {
            return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold);
        }

        private static Tensor _batched_nms_coordinate_trick(Tensor boxes, Tensor scores, Tensor idxs, float iou_threshold)
        {
            if (boxes.numel() == 0)
            {
                return torch.empty(0, dtype: torch.int64, device: boxes.device);
            }

            Tensor max_coordinate = boxes.max();
            Tensor offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes));
            Tensor boxes_for_nms = boxes + offsets[.., TensorIndex.None];
            Tensor keep = torchvision.ops.nms(boxes_for_nms, scores, iou_threshold);
            return keep;
        }

    }
    #endregion
    /// <summary>
    /// 未压缩RLE格式（对应Python uncompressed_rle）
    /// </summary>
    public class UncompressedCocoRle
    {
        /// <summary>
        /// 掩码尺寸 [height, width]（对应Python uncompressed_rle["size"]）
        /// </summary>
        public int[] Size { get; set; }

        /// <summary>
        /// 未压缩的计数列表（对应Python uncompressed_rle["counts"]）
        /// </summary>
        public List<long> Counts { get; set; }

        public UncompressedCocoRle()
        {
            Size = new int[2];
            Counts = new List<long>();
        }
    }

    /// <summary>
    /// COCO规范的压缩RLE格式（对应Python编码后的rle）
    /// </summary>
    public class CompressedCocoRle
    {
        /// <summary>
        /// 掩码尺寸 [height, width]
        /// </summary>
        public int[] Size { get; set; }

        /// <summary>
        /// 压缩后的UTF-8编码计数字符串（支持JSON序列化）
        /// </summary>
        public string Counts { get; set; }

        public CompressedCocoRle()
        {
            Size = new int[2];
            Counts = string.Empty;
        }
    }
}