using OpenCvSharp;
using Sam2Sharp.Modeling;
using System.Collections.Specialized;
using System.Diagnostics;
using Tensorboard;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SAM2Sharp
{

    // 常量定义（对应Python中的NO_OBJ_SCORE）
    public static class Constants
    {
        public const float NO_OBJ_SCORE = -1000.0f;
    }

    /// <summary>
    /// SAM2视频预测器（C#版本，OpenCvSharp替代ndarray）
    /// 保留原Python命名风格，适配C#语法和TorchSharp/OpenCvSharp
    /// </summary>
    public class SAM2VideoPredictor : Sam2Base
    {
        // 原Python构造函数参数
        private readonly float fill_hole_area;
        private readonly bool non_overlap_masks;
        private readonly bool clear_non_cond_mem_around_input;
        private readonly bool clear_non_cond_mem_for_multi_obj;
        private readonly bool add_all_frames_to_correct_as_cond;

        /// <summary>
        /// 构造函数（对应原Python __init__）
        /// </summary>
        /// <param name="fill_hole_area">填充孔洞面积阈值</param>
        /// <param name="non_overlap_masks">是否应用非重叠掩码约束</param>
        /// <param name="clear_non_cond_mem_around_input">添加修正点击后清除周围帧的非条件内存</param>
        /// <param name="clear_non_cond_mem_for_multi_obj">多对象时也清除非条件内存</param>
        /// <param name="add_all_frames_to_correct_as_cond">将所有修正帧加入条件帧列表</param>
        /// <param name="kwargs">基类参数</param>
        public SAM2VideoPredictor(
            float fill_hole_area = 0,
            bool non_overlap_masks = false,
            bool clear_non_cond_mem_around_input = false,
            bool clear_non_cond_mem_for_multi_obj = false,
            bool add_all_frames_to_correct_as_cond = false,
            Dictionary<string, object> kwargs = null) : base(kwargs)
        {
            this.fill_hole_area = fill_hole_area;
            this.non_overlap_masks = non_overlap_masks;
            this.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input;
            this.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj;
            this.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond;
        }

        /// <summary>
        /// 初始化推理状态（对应原Python init_state）
        /// </summary>
        /// <param name="video_path">视频路径</param>
        /// <param name="offload_video_to_cpu">视频帧卸载到CPU</param>
        /// <param name="offload_state_to_cpu">推理状态卸载到CPU</param>
        /// <param name="async_loading_frames">异步加载帧</param>
        /// <returns>推理状态字典</returns>
        public Dictionary<string, object> InitState(
            string video_path,
            bool offload_video_to_cpu = false,
            bool offload_state_to_cpu = false,
            bool async_loading_frames = false)
        {
            var compute_device = this.device; // 模型设备（CPU/GPU）
            Mat[] images;
            int video_height, video_width;

            // 加载视频帧（替代原load_video_frames，使用OpenCvSharp）
            LoadVideoFrames(
                video_path,
                this.image_size,
                out images,
                out video_height,
                out video_width,
                offload_video_to_cpu,
                async_loading_frames,
                compute_device);

            var inference_state = new Dictionary<string, object>
            {
                ["images"] = images,
                ["num_frames"] = images.Length,
                ["offload_video_to_cpu"] = offload_video_to_cpu,
                ["offload_state_to_cpu"] = offload_state_to_cpu,
                ["video_height"] = video_height,
                ["video_width"] = video_width,
                ["device"] = compute_device,
                ["storage_device"] = offload_state_to_cpu ? torch.CPU : compute_device,
                ["point_inputs_per_obj"] = new Dictionary<int, Dictionary<int, Tensor>>(), // obj_idx -> frame_idx -> points
                ["mask_inputs_per_obj"] = new Dictionary<int, Dictionary<int, Tensor>>(), // obj_idx -> frame_idx -> mask
                ["cached_features"] = new Dictionary<int, Dictionary<string, Tensor>>(), // frame_idx -> feature_name -> tensor
                ["constants"] = new Dictionary<string, Tensor>(),
                ["obj_id_to_idx"] = new OrderedDictionary(), // client obj_id -> model obj_idx
                ["obj_idx_to_id"] = new OrderedDictionary(), // model obj_idx -> client obj_id
                ["obj_ids"] = new List<int>(),
                ["output_dict"] = new Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>
                {
                    ["cond_frame_outputs"] = new Dictionary<int, Dictionary<string, Tensor>>(),
                    ["non_cond_frame_outputs"] = new Dictionary<int, Dictionary<string, Tensor>>()
                },
                ["output_dict_per_obj"] = new Dictionary<int, Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>>(),
                ["temp_output_dict_per_obj"] = new Dictionary<int, Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>>(),
                ["consolidated_frame_inds"] = new Dictionary<string, HashSet<int>>
                {
                    ["cond_frame_outputs"] = new HashSet<int>(),
                    ["non_cond_frame_outputs"] = new HashSet<int>()
                },
                ["tracking_has_started"] = false,
                ["frames_already_tracked"] = new Dictionary<int, Dictionary<string, bool>>() // frame_idx -> { "reverse": bool }
            };

            // 预热视觉骨干网络，缓存第0帧特征
            _GetImageFeature(inference_state, 0, 1);

            return inference_state;
        }

        /// <summary>
        /// 从预训练模型加载（对应原Python from_pretrained）
        /// </summary>
        /// <param name="model_id">HuggingFace仓库ID</param>
        /// <param name="kwargs">构造参数</param>
        /// <returns>SAM2VideoPredictor实例</returns>
        public static SAM2VideoPredictor FromPretrained(string model_id, Dictionary<string, object> kwargs = null)
        {
            // 模拟构建预训练模型（需结合实际SAM2的C#构建逻辑）
            var sam_model = BuildSAM2VideoPredictorHF(model_id, kwargs);
            return sam_model;
        }

        /// <summary>
        /// 客户端对象ID转模型对象索引（对应原Python _obj_id_to_idx）
        /// </summary>
        private int _ObjIdToIdx(Dictionary<string, object> inference_state, int obj_id)
        {
            var obj_id_to_idx = (OrderedDictionary)inference_state["obj_id_to_idx"];
            if (obj_id_to_idx.Contains(obj_id))
            {
                return (int)obj_id_to_idx[obj_id];
            }

            // 跟踪开始后不允许添加新对象
            var tracking_has_started = (bool)inference_state["tracking_has_started"];
            if (!tracking_has_started)
            {
                int obj_idx = obj_id_to_idx.Count;
                obj_id_to_idx[obj_id] = obj_idx;
                ((OrderedDictionary)inference_state["obj_idx_to_id"])[obj_idx] = obj_id;
                ((List<int>)inference_state["obj_ids"]).Add(obj_id);

                // 初始化对象相关的输入输出结构
                var point_inputs_per_obj = (Dictionary<int, Dictionary<int, Tensor>>)inference_state["point_inputs_per_obj"];
                var mask_inputs_per_obj = (Dictionary<int, Dictionary<int, Tensor>>)inference_state["mask_inputs_per_obj"];
                var output_dict_per_obj = (Dictionary<int, Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>>)inference_state["output_dict_per_obj"];
                var temp_output_dict_per_obj = (Dictionary<int, Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>>)inference_state["temp_output_dict_per_obj"];

                point_inputs_per_obj[obj_idx] = new Dictionary<int, Tensor>();
                mask_inputs_per_obj[obj_idx] = new Dictionary<int, Tensor>();
                output_dict_per_obj[obj_idx] = new Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>
                {
                    ["cond_frame_outputs"] = new Dictionary<int, Dictionary<string, Tensor>>(),
                    ["non_cond_frame_outputs"] = new Dictionary<int, Dictionary<string, Tensor>>()
                };
                temp_output_dict_per_obj[obj_idx] = new Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>
                {
                    ["cond_frame_outputs"] = new Dictionary<int, Dictionary<string, Tensor>>(),
                    ["non_cond_frame_outputs"] = new Dictionary<int, Dictionary<string, Tensor>>()
                };

                return obj_idx;
            }
            else
            {
                var obj_ids = (List<int>)inference_state["obj_ids"];
                throw new InvalidOperationException(
                    $"Cannot add new object id {obj_id} after tracking starts. " +
                    $"All existing object ids: {string.Join(", ", obj_ids)}. " +
                    "Please call 'ResetState' to restart from scratch.");
            }
        }

        /// <summary>
        /// 模型对象索引转客户端对象ID（对应原Python _obj_idx_to_id）
        /// </summary>
        private int _ObjIdxToId(Dictionary<string, object> inference_state, int obj_idx)
        {
            return (int)((OrderedDictionary)inference_state["obj_idx_to_id"])[obj_idx];
        }

        /// <summary>
        /// 获取对象数量（对应原Python _get_obj_num）
        /// </summary>
        private int _GetObjNum(Dictionary<string, object> inference_state)
        {
            return ((OrderedDictionary)inference_state["obj_idx_to_id"]).Count;
        }

        /// <summary>
        /// 添加新点或框（对应原Python add_new_points_or_box）
        /// </summary>
        public (int frame_idx, List<int> obj_ids, Tensor video_res_masks) AddNewPointsOrBox(
            Dictionary<string, object> inference_state,
            int frame_idx,
            int obj_id,
            Mat points = null,
            Mat labels = null,
            bool clear_old_points = true,
            bool normalize_coords = true,
            Mat box = null)
        {
            int obj_idx = _ObjIdToIdx(inference_state, obj_id);
            var point_inputs_per_frame = ((Dictionary<int, Dictionary<int, Tensor>>)inference_state["point_inputs_per_obj"])[obj_idx];
            var mask_inputs_per_frame = ((Dictionary<int, Dictionary<int, Tensor>>)inference_state["mask_inputs_per_obj"])[obj_idx];

            // 参数校验
            if ((points != null) != (labels != null))
                throw new ArgumentException("points and labels must be provided together");
            if (points == null && box == null)
                throw new ArgumentException("at least one of points or box must be provided as input");

            // 转换points/labels为Tensor（OpenCvSharp Mat → TorchSharp Tensor）
            Tensor points_tensor = points == null
                ? torch.zeros(0, 2, dtype: torch.float32)
                : MatToTensor(points).to(torch.float32);
            Tensor labels_tensor = labels == null
                ? torch.zeros(0, dtype: torch.int32)
                : MatToTensor(labels).to(torch.int32);

            // 添加batch维度
            if (points_tensor.dim() == 2) points_tensor = points_tensor.unsqueeze(0);
            if (labels_tensor.dim() == 1) labels_tensor = labels_tensor.unsqueeze(0);

            // 处理框输入（转换为点+标签）
            if (box != null)
            {
                if (!clear_old_points)
                    throw new ArgumentException("cannot add box without clearing old points, since " +
                        "box prompt must be provided before any point prompt (please use clear_old_points=True instead)");

                if ((bool)inference_state["tracking_has_started"])
                    Console.WriteLine("Warning: You are adding a box after tracking starts. SAM 2 may not always be " +
                        "able to incorporate a box prompt for refinement. If you intend to use box prompt as an initial input before tracking, " +
                        "please call 'ResetState' on the inference state to restart from scratch.");

                var box_tensor = MatToTensor(box).to(torch.float32).reshape(1, 2, 2);
                var box_labels = torch.tensor(new[] { 2, 3 }, dtype: torch.int32).reshape(1, 2);

                points_tensor = torch.cat(new[] { box_tensor, points_tensor }, 1);
                labels_tensor = torch.cat(new[] { box_labels, labels_tensor }, 1);
            }

            // 归一化坐标（基于视频原始分辨率）
            if (normalize_coords)
            {
                int video_H = (int)inference_state["video_height"];
                int video_W = (int)inference_state["video_width"];
                var scale = torch.tensor(new[] { video_W, video_H }, dtype: torch.float32).to(points_tensor.device);
                points_tensor = points_tensor / scale;
            }

            // 缩放到模型内部图像尺寸
            points_tensor = points_tensor * this.image_size;
            points_tensor = points_tensor.to((Device)inference_state["device"]);
            labels_tensor = labels_tensor.to((Device)inference_state["device"]);

            // 拼接旧点（如果需要）
            Tensor point_inputs = null;
            if (!clear_old_points && point_inputs_per_frame.ContainsKey(frame_idx))
                point_inputs = point_inputs_per_frame[frame_idx];

            point_inputs = ConcatPoints(point_inputs, points_tensor, labels_tensor);

            // 更新输入
            point_inputs_per_frame[frame_idx] = point_inputs;
            mask_inputs_per_frame.Remove(frame_idx);

            // 判断是否为初始条件帧
            bool is_init_cond_frame = !((Dictionary<int, Dictionary<string, bool>>)inference_state["frames_already_tracked"]).ContainsKey(frame_idx);
            bool reverse = is_init_cond_frame ? false : ((Dictionary<int, Dictionary<string, bool>>)inference_state["frames_already_tracked"])[frame_idx]["reverse"];

            // 获取输出字典
            var obj_output_dict = ((Dictionary<int, Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>>)inference_state["output_dict_per_obj"])[obj_idx];
            var obj_temp_output_dict = ((Dictionary<int, Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>>)inference_state["temp_output_dict_per_obj"])[obj_idx];
            string storage_key = (is_init_cond_frame || this.add_all_frames_to_correct_as_cond)
                ? "cond_frame_outputs"
                : "non_cond_frame_outputs";

            // 获取之前的掩码logits
            Tensor prev_sam_mask_logits = null;
            Dictionary<string, Tensor> prev_out = null;

            if (obj_temp_output_dict[storage_key].ContainsKey(frame_idx))
                prev_out = obj_temp_output_dict[storage_key][frame_idx];
            else if (obj_output_dict["cond_frame_outputs"].ContainsKey(frame_idx))
                prev_out = obj_output_dict["cond_frame_outputs"][frame_idx];
            else if (obj_output_dict["non_cond_frame_outputs"].ContainsKey(frame_idx))
                prev_out = obj_output_dict["non_cond_frame_outputs"][frame_idx];

            if (prev_out is not null && prev_out.ContainsKey("pred_masks") && prev_out["pred_masks"] is not null)
            {
                prev_sam_mask_logits = prev_out["pred_masks"].to((Device)inference_state["device"], non_blocking: true);
                prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0f, 32.0f);
            }

            // 单帧推理
            var (current_out, _) = _RunSingleFrameInference(
                inference_state,
                obj_output_dict,
                frame_idx,
                1,
                is_init_cond_frame,
                point_inputs,
                null,
                reverse,
                run_mem_encoder: false,
                prev_sam_mask_logits);

            // 更新临时输出
            if (!obj_temp_output_dict[storage_key].ContainsKey(frame_idx))
                obj_temp_output_dict[storage_key][frame_idx] = new Dictionary<string, Tensor>();
            obj_temp_output_dict[storage_key][frame_idx] = current_out;

            // 合并临时输出并缩放到原始视频分辨率
            var obj_ids = (List<int>)inference_state["obj_ids"];
            var consolidated_out = _ConsolidateTempOutputAcrossObj(
                inference_state,
                frame_idx,
                is_init_cond_frame || this.add_all_frames_to_correct_as_cond,
                run_mem_encoder: false,
                consolidate_at_video_res: true);

            var (_, video_res_masks) = _GetOrigVideoResOutput(
                inference_state,
                consolidated_out["pred_masks_video_res"]);

            return (frame_idx, obj_ids, video_res_masks);
        }

        /// <summary>
        /// 添加新掩码（对应原Python add_new_mask）
        /// </summary>
        public (int frame_idx, List<int> obj_ids, Tensor video_res_masks) AddNewMask(
            Dictionary<string, object> inference_state,
            int frame_idx,
            int obj_id,
            Mat mask)
        {
            int obj_idx = _ObjIdToIdx(inference_state, obj_id);
            var point_inputs_per_frame = ((Dictionary<int, Dictionary<int, Tensor>>)inference_state["point_inputs_per_obj"])[obj_idx];
            var mask_inputs_per_frame = ((Dictionary<int, Dictionary<int, Tensor>>)inference_state["mask_inputs_per_obj"])[obj_idx];

            // 转换掩码为Tensor（OpenCvSharp Mat → TorchSharp Tensor）
            Tensor mask_tensor = MatToTensor(mask).to(torch.float32);
            Debug.Assert(mask_tensor.dim() == 2);

            // 添加batch和channel维度
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0);
            mask_tensor = mask_tensor.to((Device)inference_state["device"]);

            // 调整掩码尺寸到模型输入尺寸
            int mask_H = mask.Rows;
            int mask_W = mask.Cols;
            if (mask_H != this.image_size || mask_W != this.image_size)
            {
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor,
                    size: new long[] { this.image_size, this.image_size },
                    mode: InterpolationMode.Bilinear,
                    align_corners: false,
                    antialias: true);
                mask_tensor = (mask_tensor >= 0.5f).to(torch.float32);
            }

            // 更新输入
            mask_inputs_per_frame[frame_idx] = mask_tensor;
            point_inputs_per_frame.Remove(frame_idx);

            // 判断是否为初始条件帧
            bool is_init_cond_frame = !((Dictionary<int, Dictionary<string, bool>>)inference_state["frames_already_tracked"]).ContainsKey(frame_idx);
            bool reverse = is_init_cond_frame ? false : ((Dictionary<int, Dictionary<string, bool>>)inference_state["frames_already_tracked"])[frame_idx]["reverse"];

            // 获取输出字典
            var obj_output_dict = ((Dictionary<int, Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>>)inference_state["output_dict_per_obj"])[obj_idx];
            var obj_temp_output_dict = ((Dictionary<int, Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>>)inference_state["temp_output_dict_per_obj"])[obj_idx];
            string storage_key = (is_init_cond_frame || this.add_all_frames_to_correct_as_cond)
                ? "cond_frame_outputs"
                : "non_cond_frame_outputs";

            // 单帧推理
            var (current_out, _) = _RunSingleFrameInference(
                inference_state,
                obj_output_dict,
                frame_idx,
                1,
                is_init_cond_frame,
                null,
                mask_tensor,
                reverse,
                run_mem_encoder: false);

            // 更新临时输出
            if (!obj_temp_output_dict[storage_key].ContainsKey(frame_idx))
                obj_temp_output_dict[storage_key][frame_idx] = new Dictionary<string, Tensor>();
            obj_temp_output_dict[storage_key][frame_idx] = current_out;

            // 合并临时输出并缩放到原始视频分辨率
            var obj_ids = (List<int>)inference_state["obj_ids"];
            var consolidated_out = _ConsolidateTempOutputAcrossObj(
                inference_state,
                frame_idx,
                is_init_cond_frame || this.add_all_frames_to_correct_as_cond,
                run_mem_encoder: false,
                consolidate_at_video_res: true);

            var (_, video_res_masks) = _GetOrigVideoResOutput(
                inference_state,
                consolidated_out["pred_masks_video_res"]);

            return (frame_idx, obj_ids, video_res_masks);
        }

        // ------------------------------ 私有辅助方法 ------------------------------
        /// <summary>
        /// OpenCvSharp Mat转TorchSharp Tensor
        /// </summary>
        private Tensor MatToTensor(Mat mat)
        {
            // 处理不同数据类型的Mat转换
            Tensor tensor;
            if(mat.Type()== MatType.CV_32F)
                tensor = torch.empty([mat.Rows, mat.Cols], torch.float32);
            else if(mat.Type() == MatType.CV_32SC1)
                tensor = torch.empty([mat.Rows, mat.Cols], torch.float32);
            var dtype = mat.Type() switch
                {
                    MatType.CV_32F => torch.float32,
                    MatType.CV_64F => torch.float64,
                    MatType.CV_32S => torch.int32,
                    MatType.CV_8U => torch.uint8,
                    _ => torch.float32
                };

            return torch.from_numpy(mat.ToBytes(), dtype: dtype)
                .reshape(mat.Rows, mat.Cols, mat.Channels())
                .permute(2, 0, 1) // HWC → CHW
                .contiguous();
        }
        static OpenCvSharp.MatType[][] cmt = new OpenCvSharp.MatType[6][]{
        [MatType.CV_32FC1 , MatType.CV_32FC2 , MatType.CV_32FC3 , MatType.CV_32FC4 ],
        [MatType.CV_64FC1 , MatType.CV_64FC2 , MatType.CV_64FC3 , MatType.CV_64FC4 ],
        [MatType.CV_32SC1 , MatType.CV_32SC2 , MatType.CV_32SC3 , MatType.CV_32SC4 ],
        [MatType.CV_8UC1 , MatType.CV_8UC2 , MatType.CV_8UC3 , MatType.CV_8UC4 ],
        [MatType.CV_16UC1 ],
        [MatType.CV_16SC1 ] };

        /// <summary>
        /// 加载视频帧（替代原Python load_video_frames，使用OpenCvSharp）
        /// </summary>
        private void LoadVideoFrames(
            string video_path,
            int image_size,
            out Mat[] images,
            out int video_height,
            out int video_width,
            bool offload_video_to_cpu,
            bool async_loading_frames,
            Device compute_device)
        {
            using var capture = new VideoCapture(video_path);
            if (!capture.IsOpened())
                throw new FileNotFoundException("Video file not found or cannot be opened", video_path);

            video_height = (int)capture.FrameHeight;
            video_width = (int)capture.FrameWidth;

            var frameList = new List<Mat>();
            Mat frame = new Mat();

            // 异步加载（简化实现）
            if (async_loading_frames)
            {
                // 实际场景需实现异步读取逻辑
                while (capture.Read(frame))
                {
                    var resized = ResizeFrame(frame, image_size);
                    frameList.Add(offload_video_to_cpu ? resized.Clone() : resized);
                }
            }
            else
            {
                while (capture.Read(frame))
                {
                    var resized = ResizeFrame(frame, image_size);
                    frameList.Add(offload_video_to_cpu ? resized.Clone() : resized);
                }
            }

            images = frameList.ToArray();
        }

        /// <summary>
        /// 调整帧尺寸（保持比例，缩放到image_size）
        /// </summary>
        private Mat ResizeFrame(Mat frame, int image_size)
        {
            double scale = Math.Min((double)image_size / frame.Width, (double)image_size / frame.Height);
            int newW = (int)(frame.Width * scale);
            int newH = (int)(frame.Height * scale);
            Cv2.Resize(frame, frame, new OpenCvSharp.Size(newW, newH), interpolation: InterpolationFlags.Linear);
            return frame;
        }

        /// <summary>
        /// 拼接点（对应原Python concat_points）
        /// </summary>
        private Tensor ConcatPoints(Tensor prev_points, Tensor new_points, Tensor new_labels)
        {
            if (prev_points is null)
                return torch.cat(new[] { new_points, new_labels.unsqueeze(-1) }, -1);

            // 修复核心错误：替换Python省略号索引为TorchSharp的narrow方法
            // prev_points[..., :2] → 取最后两个维度的前2列（点坐标）
            var prev_pts = prev_points.narrow(-1, 0, 2);
            // prev_points[..., 2] → 取最后一维的第2个元素（标签）
            var prev_lbls = prev_points.narrow(-1, 2, 1).squeeze(-1);

            // 拼接新点和历史点
            var concat_pts = torch.cat(new[] { prev_pts, new_points }, 1);
            var concat_lbls = torch.cat(new[] { prev_lbls, new_labels }, 1);

            // 合并坐标和标签（最后一维拼接）
            return torch.cat(new[] { concat_pts, concat_lbls.unsqueeze(-1) }, -1);
        }

        /// <summary>
        /// 缩放到原始视频分辨率并应用非重叠约束（对应原Python _get_orig_video_res_output）
        /// </summary>
        private (Tensor any_res_masks, Tensor video_res_masks) _GetOrigVideoResOutput(
            Dictionary<string, object> inference_state,
            Tensor any_res_masks)
        {
            var device = (Device)inference_state["device"];
            int video_H = (int)inference_state["video_height"];
            int video_W = (int)inference_state["video_width"];

            any_res_masks = any_res_masks.to(device, non_blocking: true);
            Tensor video_res_masks;

            if (any_res_masks.shape[^2] == video_H && any_res_masks.shape[^1] == video_W)
            {
                video_res_masks = any_res_masks;
            }
            else
            {
                video_res_masks = torch.nn.functional.interpolate(
                    any_res_masks,
                    size: new long[] { video_H, video_W },
                    mode: InterpolationMode.Bilinear,
                    align_corners: false);
            }

            // 应用非重叠掩码约束
            if (this.non_overlap_masks)
            {
                video_res_masks = _ApplyNonOverlappingConstraints(video_res_masks);
            }

            return (any_res_masks, video_res_masks);
        }

        /// <summary>
        /// 应用非重叠掩码约束（核心逻辑需根据SAM2论文实现）
        /// </summary>
        private Tensor _ApplyNonOverlappingConstraints(Tensor masks)
        {
            // 简化实现：对每个像素，只保留得分最高的对象掩码
            var (max_vals, max_indices) = torch.max(masks, 0);
            var one_hot = torch.nn.functional.one_hot(max_indices, masks.shape[0])
                .permute(3, 0, 1, 2)
                .to(masks.dtype);
            return masks * one_hot;
        }

        /// <summary>
        /// 合并多对象临时输出（对应原Python _consolidate_temp_output_across_obj）
        /// </summary>
        private Dictionary<string, Tensor> _ConsolidateTempOutputAcrossObj(
            Dictionary<string, object> inference_state,
            int frame_idx,
            bool is_cond,
            bool run_mem_encoder,
            bool consolidate_at_video_res = false)
        {
            int batch_size = _GetObjNum(inference_state);
            string storage_key = is_cond ? "cond_frame_outputs" : "non_cond_frame_outputs";

            // 确定合并后的分辨率
            int consolidated_H, consolidated_W;
            string consolidated_mask_key;

            if (consolidate_at_video_res)
            {
                Debug.Assert(!run_mem_encoder, "memory encoder cannot run at video resolution");
                consolidated_H = (int)inference_state["video_height"];
                consolidated_W = (int)inference_state["video_width"];
                consolidated_mask_key = "pred_masks_video_res";
            }
            else
            {
                consolidated_H = consolidated_W = this.image_size / 4;
                consolidated_mask_key = "pred_masks";
            }

            // 初始化合并输出
            var consolidated_out = new Dictionary<string, Tensor>
            {
                ["maskmem_features"] = null,
                ["maskmem_pos_enc"] = null,
                [consolidated_mask_key] = torch.full(
                    size: new long[] { batch_size, 1, consolidated_H, consolidated_W },
                    value: Constants.NO_OBJ_SCORE,
                    dtype: torch.float32,
                    device: (Device)inference_state["storage_device"]),
                ["obj_ptr"] = torch.full(
                    size: new[] { batch_size, this.hidden_dim },
                    fill_value: Constants.NO_OBJ_SCORE,
                    dtype: torch.float32,
                    device: (Device)inference_state["device"]),
                ["object_score_logits"] = torch.full(
                    size: new long[] { batch_size, 1 },
                    value: 10.0f, // 默认得分（sigmoid(10)≈1）
                    dtype: torch.float32,
                    device: (Device)inference_state["device"])
            };

            Tensor empty_mask_ptr = null;

            // 遍历所有对象
            for (int obj_idx = 0; obj_idx < batch_size; obj_idx++)
            {
                var obj_temp_output_dict = ((Dictionary<int, Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>>)inference_state["temp_output_dict_per_obj"])[obj_idx];
                var obj_output_dict = ((Dictionary<int, Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>>)inference_state["output_dict_per_obj"])[obj_idx];

                Dictionary<string, Tensor> out_dict = null;
                if (obj_temp_output_dict[storage_key].ContainsKey(frame_idx))
                    out_dict = obj_temp_output_dict[storage_key][frame_idx];
                else if (obj_output_dict["cond_frame_outputs"].ContainsKey(frame_idx))
                    out_dict = obj_output_dict["cond_frame_outputs"][frame_idx];
                else if (obj_output_dict["non_cond_frame_outputs"].ContainsKey(frame_idx))
                    out_dict = obj_output_dict["non_cond_frame_outputs"][frame_idx];

                // 无输出时填充空指针
                if (out_dict == null)
                {
                    if (run_mem_encoder)
                    {
                        if (empty_mask_ptr is null)
                            empty_mask_ptr = _GetEmptyMaskPtr(inference_state, frame_idx);
                        consolidated_out["obj_ptr"][obj_idx] = empty_mask_ptr;
                    }
                    continue;
                }

                // 合并掩码
                var obj_mask = out_dict["pred_masks"];
                var consolidated_pred_masks = consolidated_out[consolidated_mask_key];

                if (obj_mask.shape[^2] == consolidated_pred_masks.shape[^2] && obj_mask.shape[^1] == consolidated_pred_masks.shape[^1])
                {
                    consolidated_pred_masks[obj_idx] = obj_mask;
                }
                else
                {
                    var resized_obj_mask = torch.nn.functional.interpolate(
                        obj_mask,
                        size: new[] { consolidated_pred_masks.shape[^2], consolidated_pred_masks.shape[^1] },
                        mode: InterpolationMode.Bilinear,
                        align_corners: false);
                    consolidated_pred_masks[obj_idx] = resized_obj_mask;
                }

                // 合并对象指针和得分
                consolidated_out["obj_ptr"][obj_idx] = out_dict["obj_ptr"];
                consolidated_out["object_score_logits"][obj_idx] = out_dict["object_score_logits"];
            }

            // 运行内存编码器（如果需要）
            if (run_mem_encoder)
            {
                var device = (Device)inference_state["device"];
                var high_res_masks = torch.nn.functional.interpolate(
                    consolidated_out["pred_masks"].to(device, non_blocking: true),
                    size: new long[] { this.image_size, this.image_size },
                    mode: InterpolationMode.Bilinear,
                    align_corners: false);

                if (this.non_overlap_masks)
                    high_res_masks = _ApplyNonOverlappingConstraints(high_res_masks);

                var (maskmem_features, maskmem_pos_enc) = _RunMemoryEncoder(
                    inference_state,
                    frame_idx,
                    batch_size,
                    high_res_masks,
                    consolidated_out["object_score_logits"],
                    is_mask_from_pts: true);

                consolidated_out["maskmem_features"] = maskmem_features;
                consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc;
            }

            return consolidated_out;
        }

        /// <summary>
        /// 获取空掩码指针（对应原Python _get_empty_mask_ptr）
        /// </summary>
        private Tensor _GetEmptyMaskPtr(Dictionary<string, object> inference_state, int frame_idx)
        {
            int batch_size = 1;
            var mask_inputs = torch.zeros(
                new long[] { batch_size, 1, this.image_size, this.image_size },
                dtype: torch.float32,
                device: (Device)inference_state["device"]);

            // 获取图像特征
            var (_, _, current_vision_feats, current_vision_pos_embeds, feat_sizes) = _GetImageFeature(
                inference_state, frame_idx, batch_size);

            // 运行跟踪步骤获取空指针
            var current_out = TrackStep(
                frame_idx: frame_idx,
                is_init_cond_frame: true,
                current_vision_feats: current_vision_feats,
                current_vision_pos_embeds: current_vision_pos_embeds,
                feat_sizes: feat_sizes,
                point_inputs: null,
                mask_inputs: mask_inputs,
                output_dict: new Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>(),
                num_frames: (int)inference_state["num_frames"],
                track_in_reverse: false,
                run_mem_encoder: false,
                prev_sam_mask_logits: null);

            return current_out["obj_ptr"];
        }

        // ------------------------------ 未实现的占位方法（需结合SAM2Base补充） ------------------------------
        private (Tensor, Tensor, Tensor, Tensor, Dictionary<string, int>) _GetImageFeature(
            Dictionary<string, object> inference_state, int frame_idx, int batch_size)
        {
            // 需结合SAM2Base的视觉特征提取逻辑实现
            throw new NotImplementedException();
        }

        private (Dictionary<string, Tensor>, Tensor) _RunSingleFrameInference(
            Dictionary<string, object> inference_state,
            Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>> output_dict,
            int frame_idx,
            int batch_size,
            bool is_init_cond_frame,
            Tensor point_inputs,
            Tensor mask_inputs,
            bool reverse,
            bool run_mem_encoder,
            Tensor prev_sam_mask_logits = null)
        {
            // 需结合SAM2的单帧推理逻辑实现
            throw new NotImplementedException();
        }

        private (Tensor, Tensor) _RunMemoryEncoder(
            Dictionary<string, object> inference_state,
            int frame_idx,
            int batch_size,
            Tensor high_res_masks,
            Tensor object_score_logits,
            bool is_mask_from_pts)
        {
            // 需结合SAM2的内存编码器逻辑实现
            throw new NotImplementedException();
        }

        private Dictionary<string, Tensor> TrackStep(
            int frame_idx,
            bool is_init_cond_frame,
            Tensor current_vision_feats,
            Tensor current_vision_pos_embeds,
            Dictionary<string, int> feat_sizes,
            Tensor point_inputs,
            Tensor mask_inputs,
            Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>> output_dict,
            int num_frames,
            bool track_in_reverse,
            bool run_mem_encoder,
            Tensor prev_sam_mask_logits = null)
        {
            // 需结合SAM2的跟踪步骤逻辑实现
            throw new NotImplementedException();
        }

        private static SAM2VideoPredictor BuildSAM2VideoPredictorHF(string model_id, Dictionary<string, object> kwargs)
        {
            // 需结合HuggingFace模型加载逻辑实现
            throw new NotImplementedException();
        }

        private void _ClearNonCondMemAroundInput(Dictionary<string, object> inference_state, int frame_idx)
        {
            // 需实现清除周围帧非条件内存的逻辑
            throw new NotImplementedException();
        }

        private void _AddOutputPerObject(
            Dictionary<string, object> inference_state,
            int frame_idx,
            Dictionary<string, Tensor> consolidated_out,
            string storage_key)
        {
            // 需实现将合并输出拆分到每个对象的逻辑
            throw new NotImplementedException();
        }

        /// <summary>
        /// 跟踪前预处理（对应原Python propagate_in_video_preflight）
        /// </summary>
        public void PropagateInVideoPreflight(Dictionary<string, object> inference_state)
        {
            inference_state["tracking_has_started"] = true;
            int batch_size = _GetObjNum(inference_state);

            var temp_output_dict_per_obj = (Dictionary<int, Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>>)inference_state["temp_output_dict_per_obj"];
            var output_dict = (Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>)inference_state["output_dict"];
            var consolidated_frame_inds = (Dictionary<string, HashSet<int>>)inference_state["consolidated_frame_inds"];

            // 合并条件/非条件临时输出
            foreach (var is_cond in new[] { false, true })
            {
                string storage_key = is_cond ? "cond_frame_outputs" : "non_cond_frame_outputs";
                var temp_frame_inds = new HashSet<int>();

                // 收集所有有临时输出的帧
                foreach (var obj_temp_output_dict in temp_output_dict_per_obj.Values)
                {
                    foreach (var frame_idx in obj_temp_output_dict[storage_key].Keys)
                        temp_frame_inds.Add(frame_idx);
                }

                // 更新已合并帧索引
                consolidated_frame_inds[storage_key].UnionWith(temp_frame_inds);

                // 合并每个帧的输出
                foreach (var frame_idx in temp_frame_inds)
                {
                    var consolidated_out = _ConsolidateTempOutputAcrossObj(
                        inference_state, frame_idx, is_cond, run_mem_encoder: true);

                    output_dict[storage_key][frame_idx] = consolidated_out;
                    _AddOutputPerObject(inference_state, frame_idx, consolidated_out, storage_key);

                    // 清除非条件内存（如果需要）
                    bool clear_non_cond_mem = this.clear_non_cond_mem_around_input &&
                        (this.clear_non_cond_mem_for_multi_obj || batch_size <= 1);
                    if (clear_non_cond_mem)
                        _ClearNonCondMemAroundInput(inference_state, frame_idx);
                }

                // 清空临时输出
                foreach (var obj_temp_output_dict in temp_output_dict_per_obj.Values)
                    obj_temp_output_dict[storage_key].Clear();
            }

            // 移除重复的非条件帧输出
            foreach (var frame_idx in output_dict["cond_frame_outputs"].Keys)
            {
                output_dict["non_cond_frame_outputs"].Remove(frame_idx);
                foreach (var obj_output_dict in ((Dictionary<int, Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>>)inference_state["output_dict_per_obj"]).Values)
                    obj_output_dict["non_cond_frame_outputs"].Remove(frame_idx);
                consolidated_frame_inds["non_cond_frame_outputs"].Remove(frame_idx);
            }

            // 验证合并帧索引与输入帧索引一致
            var all_consolidated_frame_inds = new HashSet<int>(consolidated_frame_inds["cond_frame_outputs"]);
            all_consolidated_frame_inds.UnionWith(consolidated_frame_inds["non_cond_frame_outputs"]);

            var input_frames_inds = new HashSet<int>();
            foreach (var point_inputs_per_frame in ((Dictionary<int, Dictionary<int, Tensor>>)inference_state["point_inputs_per_obj"]).Values)
                input_frames_inds.UnionWith(point_inputs_per_frame.Keys);
            foreach (var mask_inputs_per_frame in ((Dictionary<int, Dictionary<int, Tensor>>)inference_state["mask_inputs_per_obj"]).Values)
                input_frames_inds.UnionWith(mask_inputs_per_frame.Keys);

            Debug.Assert(all_consolidated_frame_inds.SetEquals(input_frames_inds), "Consolidated frame indices do not match input frame indices");
        }

        /// <summary>
        /// 视频中传播跟踪（对应原Python propagate_in_video）
        /// </summary>
        public void PropagateInVideo(
            Dictionary<string, object> inference_state,
            int? start_frame_idx = null,
            int? max_frame_num_to_track = null,
            bool reverse = false)
        {
            PropagateInVideoPreflight(inference_state);

            var output_dict = (Dictionary<string, Dictionary<int, Dictionary<string, Tensor>>>)inference_state["output_dict"];
            var consolidated_frame_inds = (Dictionary<string, HashSet<int>>)inference_state["consolidated_frame_inds"];
            int num_frames = (int)inference_state["num_frames"];
            int batch_size = _GetObjNum(inference_state);

            if (output_dict["cond_frame_outputs"].Count == 0)
                throw new InvalidOperationException("No points are provided; please add points first");

            bool clear_non_cond_mem = this.clear_non_cond_mem_around_input &&
                (this.clear_non_cond_mem_for_multi_obj || batch_size <= 1);

            // 确定起始/结束帧和处理顺序
            int start_idx = start_frame_idx ?? output_dict["cond_frame_outputs"].Keys.Min();
            int max_frames = max_frame_num_to_track ?? num_frames;
            int end_idx;
            IEnumerable<int> processing_order;

            if (reverse)
            {
                end_idx = Math.Max(start_idx - max_frames, 0);
                processing_order = start_idx > 0
                    ? Enumerable.Range(end_idx, start_idx - end_idx + 1).Reverse()
                    : Enumerable.Empty<int>();
            }
            else
            {
                end_idx = Math.Min(start_idx + max_frames, num_frames - 1);
                processing_order = Enumerable.Range(start_idx, end_idx - start_idx + 1);
            }

            // 进度条（替代TQDM）
            var progressBar = new ProgressBar();
            foreach (var frame_idx in processing_order)
            {
                progressBar.Report((double)(frame_idx - start_idx) / (end_idx - start_idx));

                if (consolidated_frame_inds["cond_frame_outputs"].Contains(frame_idx))
                {
                    string storage_key = "cond_frame_outputs";
                    var current_out = output_dict[storage_key][frame_idx];
                    var pred_masks = current_out["pred_masks"];

                    if (clear_non_cond_mem)
                        _ClearNonCondMemAroundInput(inference_state, frame_idx);
                }
                else if (consolidated_frame_inds["non_cond_frame_outputs"].Contains(frame_idx))
                {
                    string storage_key = "non_cond_frame_outputs";
                    var current_out = output_dict[storage_key][frame_idx];
                    var pred_masks = current_out["pred_masks"];
                }
                else
                {
                    string storage_key = "non_cond_frame_outputs";
                    var (current_out, pred_masks) = _RunSingleFrameInference(
                        inference_state,
                        output_dict,
                        frame_idx,
                        batch_size,
                        is_init_cond_frame: false,
                        point_inputs: null,
                        mask_inputs: null,
                        reverse: reverse,
                        run_mem_encoder: true);

                    output_dict[storage_key][frame_idx] = current_out;
                    _AddOutputPerObject(inference_state, frame_idx, current_out, storage_key);
                    ((Dictionary<int, Dictionary<string, bool>>)inference_state["frames_already_tracked"])[frame_idx] = new Dictionary<string, bool>
                    {
                        ["reverse"] = reverse
                    };
                }
            }
            progressBar.Finish();
        }
    }



    /// <summary>
    /// 简易进度条（替代Python TQDM）
    /// </summary>
    public class ProgressBar : IDisposable
    {
        private const int blockCount = 10;
        private readonly DateTime startTime = DateTime.Now;

        public void Report(double progress)
        {
            progress = Math.Clamp(progress, 0, 1);
            int completedBlocks = (int)(progress * blockCount);
            int remainingBlocks = blockCount - completedBlocks;

            Console.Write("\r[{0}{1}] {2:P2} ({3})",
                new string('#', completedBlocks),
                new string('-', remainingBlocks),
                progress,
                DateTime.Now - startTime);
        }

        public void Finish()
        {
            Report(1);
            Console.WriteLine();
        }

        public void Dispose()
        {
            Console.WriteLine();
        }
    }
}