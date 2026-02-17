using OpenCvSharp;
using Sam2Sharp.Modeling;
using System.Collections.Generic;
using System.Collections.Specialized;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Size = OpenCvSharp.Size;
using Scalar = OpenCvSharp.Scalar;
namespace SAM2Sharp
{
            /// <summary>
            /// </summary>
            public class SAM2VideoPredictor
            {
                private float _fillHoleArea;
                private bool _nonOverlapMasks;
                private bool _clearNonCondMemAroundInput;
                private bool _addAllFramesToCorrectAsCond;
                private Sam2Base model;

                /// <summary>
                /// </summary>
                /// <param name="fillHoleArea">原fill_hole_area</param>
                /// <param name="nonOverlapMasks">原non_overlap_masks</param>
                /// <param name="clearNonCondMemAroundInput">原clear_non_cond_mem_around_input</param>
                /// <param name="addAllFramesToCorrectAsCond">原add_all_frames_to_correct_as_cond</param>
                /// <param name="kwargs">扩展参数（C#用字典模拟）</param>
                public SAM2VideoPredictor(
                    float fillHoleArea = 0,
                    bool nonOverlapMasks = false,
                    bool clearNonCondMemAroundInput = false,
                    bool addAllFramesToCorrectAsCond = false,
                    Dictionary<string, object> kwargs = null) 
                {
                    _fillHoleArea = fillHoleArea;
                    _nonOverlapMasks = nonOverlapMasks;
                    _clearNonCondMemAroundInput = clearNonCondMemAroundInput;
                    _addAllFramesToCorrectAsCond = addAllFramesToCorrectAsCond;
                }

                /// <summary>
                /// 推理状态容器，对应原inference_state字典
                /// </summary>
                public class InferenceState
                {
                    // 视频帧（OpenCvSharp的Mat替代ndarray）
                    public List<Mat> Images { get; set; } = new List<Mat>();
                    public int NumFrames => Images.Count;
                    public bool OffloadVideoToCpu { get; set; }
                    public bool OffloadStateToCpu { get; set; }
                    public int VideoHeight { get; set; }
                    public int VideoWidth { get; set; }
                    public string Device { get; set; } = "cpu"; // 简化：仅标记设备，无实际GPU管理
                    public string StorageDevice { get; set; } = "cpu";

                    // 每个对象的点输入
                    public System.Collections.Specialized.OrderedDictionary PointInputsPerObj { get; set; } = new OrderedDictionary();
                    // 每个对象的掩码输入
                    public OrderedDictionary MaskInputsPerObj { get; set; } = new OrderedDictionary();
                    // 缓存特征
                    public Dictionary<int, Mat> CachedFeatures { get; set; } = new Dictionary<int, Mat>();
                    // 常量值
                    public Dictionary<string, object> Constants { get; set; } = new Dictionary<string, object>();
                    // 对象ID映射
                    public OrderedDictionary ObjIdToIdx { get; set; } = new OrderedDictionary();
                    public OrderedDictionary ObjIdxToId { get; set; } = new OrderedDictionary();
                    public List<object> ObjIds { get; set; } = new List<object>();
                    // 每个对象的输出字典
                    public OrderedDictionary OutputDictPerObj { get; set; } = new OrderedDictionary();
                    // 临时输出字典
                    public OrderedDictionary TempOutputDictPerObj { get; set; } = new OrderedDictionary();
                    // 已跟踪的帧
                    public OrderedDictionary FramesTrackedPerObj { get; set; } = new OrderedDictionary();
                }

                /// <summary>
                /// 初始化推理状态，对应原init_state方法
                /// </summary>
                /// <param name="videoPath">视频路径</param>
                /// <param name="offloadVideoToCpu">是否卸载视频到CPU</param>
                /// <param name="offloadStateToCpu">是否卸载状态到CPU</param>
                /// <param name="asyncLoadingFrames">是否异步加载帧</param>
                /// <returns>推理状态</returns>
                public InferenceState InitState(
                    string videoPath,
                    bool offloadVideoToCpu = false,
                    bool offloadStateToCpu = false,
                    bool asyncLoadingFrames = false)
                {
                    var inferenceState = new InferenceState();
                    inferenceState.OffloadVideoToCpu = offloadVideoToCpu;
                    inferenceState.OffloadStateToCpu = offloadStateToCpu;

                    // 加载视频帧（OpenCvSharp替代原load_video_frames）
                    using (var capture = new VideoCapture(videoPath))
                    {
                        if (!capture.IsOpened())
                            throw new InvalidOperationException("无法打开视频文件");

                        inferenceState.VideoHeight = (int)capture.FrameHeight;
                        inferenceState.VideoWidth = (int)capture.FrameWidth;

                        // 读取所有帧并缩放到模型输入尺寸
                        Mat frame = new Mat();
                        while (capture.Read(frame))
                        {
                            Mat resizedFrame = new Mat();
                            Cv2.Resize(frame, resizedFrame, new Size(model.image_size, model.image_size));
                            inferenceState.Images.Add(resizedFrame.Clone());
                            frame.Release();
                        }
                    }

                    inferenceState.StorageDevice = offloadStateToCpu ? "cpu" : inferenceState.Device;

                    // 初始化集合
                    inferenceState.PointInputsPerObj = new OrderedDictionary();
                    inferenceState.MaskInputsPerObj = new OrderedDictionary();
                    inferenceState.OutputDictPerObj = new OrderedDictionary();
                    inferenceState.TempOutputDictPerObj = new OrderedDictionary();
                    inferenceState.FramesTrackedPerObj = new OrderedDictionary();

                    // 预热视觉骨干网络，缓存第0帧特征
                    _GetImageFeature(inferenceState, 0, 1);

                    return inferenceState;
                }

                /// <summary>
                /// 从预训练模型加载，对应原from_pretrained方法（C#简化：仅保留接口）
                /// </summary>
                /// <param name="modelId">模型ID</param>
                /// <param name="kwargs">扩展参数</param>
                /// <returns>预测器实例</returns>
                public static SAM2VideoPredictor FromPretrained(string modelId, Dictionary<string, object> kwargs = null)
                {
                    // 原逻辑：构建HF模型 → 此处简化为直接创建实例
                    var predictor = new SAM2VideoPredictor(
                        fillHoleArea: kwargs.ContainsKey("fill_hole_area") ? (float)kwargs["fill_hole_area"] : 0,
                        nonOverlapMasks: kwargs.ContainsKey("non_overlap_masks") ? (bool)kwargs["non_overlap_masks"] : false,
                        clearNonCondMemAroundInput: kwargs.ContainsKey("clear_non_cond_mem_around_input") ? (bool)kwargs["clear_non_cond_mem_around_input"] : false,
                        addAllFramesToCorrectAsCond: kwargs.ContainsKey("add_all_frames_to_correct_as_cond") ? (bool)kwargs["add_all_frames_to_correct_as_cond"] : false,
                        kwargs: kwargs
                    );
                    return predictor;
                }

                /// <summary>
                /// 对象ID转索引，对应原_obj_id_to_idx方法
                /// </summary>
                private int _ObjIdToIdx(InferenceState inferenceState, object objId)
                {
                    if (inferenceState.ObjIdToIdx.Contains(objId))
                        return (int)inferenceState.ObjIdToIdx[objId];

                    // 新增对象
                    int objIdx = inferenceState.ObjIdToIdx.Count;
                    inferenceState.ObjIdToIdx.Add(objId, objIdx);
                    inferenceState.ObjIdxToId.Add(objIdx, objId);
                    inferenceState.ObjIds.Add(objId);

                    // 初始化对象输入输出结构
                    inferenceState.PointInputsPerObj.Add(objIdx, new Dictionary<int, Mat>());
                    inferenceState.MaskInputsPerObj.Add(objIdx, new Dictionary<int, Mat>());
                    inferenceState.OutputDictPerObj.Add(objIdx, new Dictionary<string, Dictionary<int, object>>
                    {
                        ["cond_frame_outputs"] = new Dictionary<int, object>(),
                        ["non_cond_frame_outputs"] = new Dictionary<int, object>()
                    });
                    inferenceState.TempOutputDictPerObj.Add(objIdx, new Dictionary<string, Dictionary<int, object>>
                    {
                        ["cond_frame_outputs"] = new Dictionary<int, object>(),
                        ["non_cond_frame_outputs"] = new Dictionary<int, object>()
                    });
                    inferenceState.FramesTrackedPerObj.Add(objIdx, new Dictionary<int, Dictionary<string, bool>>());

                    return objIdx;
                }

                /// <summary>
                /// 对象索引转ID，对应原_obj_idx_to_id方法
                /// </summary>
                private object _ObjIdxToId(InferenceState inferenceState, int objIdx)
                {
                    return inferenceState.ObjIdxToId[objIdx];
                }

                /// <summary>
                /// 获取对象数量，对应原_get_obj_num方法
                /// </summary>
                private int _GetObjNum(InferenceState inferenceState)
                {
                    return inferenceState.ObjIdxToId.Count;
                }

                /// <summary>
                /// 添加新点或框，对应原add_new_points_or_box方法
                /// </summary>
                /// <param name="inferenceState">推理状态</param>
                /// <param name="frameIdx">帧索引</param>
                /// <param name="objId">对象ID</param>
                /// <param name="points">点坐标（OpenCvSharp.Point2f数组）</param>
                /// <param name="labels">标签</param>
                /// <param name="clearOldPoints">是否清除旧点</param>
                /// <param name="normalizeCoords">是否归一化坐标</param>
                /// <param name="box">框（x1,y1,x2,y2）</param>
                /// <returns>(帧索引, 对象ID列表, 视频分辨率掩码)</returns>
                public (int, List<object>, Mat) AddNewPointsOrBox(
                    InferenceState inferenceState,
                    int frameIdx,
                    object objId,
                    Point2f[] points = null,
                    int[] labels = null,
                    bool clearOldPoints = true,
                    bool normalizeCoords = true,
                    Rect? box = null)
                {
                    int objIdx = _ObjIdToIdx(inferenceState, objId);
                    var pointInputsPerFrame = (Dictionary<int, Mat>)inferenceState.PointInputsPerObj[objIdx];
                    var maskInputsPerFrame = (Dictionary<int, Mat>)inferenceState.MaskInputsPerObj[objIdx];

                    // 参数校验
                    if ((points != null) != (labels != null))
                        throw new ArgumentException("points和labels必须同时提供");
                    if (points == null && box == null)
                        throw new ArgumentException("必须提供points或box");

                    // 初始化点和标签（转换为OpenCvSharp Mat替代torch.Tensor）
                    Mat pointsMat = new Mat();
                    Mat labelsMat = new Mat();
                    if (points == null)
                    {
                        pointsMat = new Mat(0, 2, MatType.CV_32F);
                        labelsMat = new Mat(0, 1, MatType.CV_32S);
                    }
                    else
                    {
                        // 转换Point2f数组到Mat
                        pointsMat = new Mat(points.Length, 2, MatType.CV_32F);
                        for (int i = 0; i < points.Length; i++)
                        {
                            pointsMat.Set<float>(i, 0, points[i].X);
                            pointsMat.Set<float>(i, 1, points[i].Y);
                        }
                        // 转换标签到Mat
                        labelsMat = new Mat(labels.Length, 1, MatType.CV_32S);
                        for (int i = 0; i < labels.Length; i++)
                        {
                            labelsMat.Set<int>(i, 0, labels[i]);
                        }
                        // 添加batch维度
                        pointsMat = pointsMat.Reshape(1, 1);
                        labelsMat = labelsMat.Reshape(1, 1);
                    }

                    // 处理框输入（转换为SAM2的点格式）
                    if (box.HasValue)
                    {
                        if (!clearOldPoints)
                            throw new ArgumentException("添加框时必须清除旧点（clearOldPoints=True）");

                        // 框转换为两个点（标签2/3）
                        var boxPoints = new[]
                        {
                    new Point2f(box.Value.X, box.Value.Y),
                    new Point2f(box.Value.X + box.Value.Width, box.Value.Y + box.Value.Height)
                };
                        var boxLabels = new[] { 2, 3 };

                        // 转换为Mat并拼接
                        Mat boxPointsMat = new Mat(1, 2, MatType.CV_32F);
                        boxPointsMat.Set<float>(0, 0, boxPoints[0].X);
                        boxPointsMat.Set<float>(0, 1, boxPoints[0].Y);
                        boxPointsMat.Set<float>(1, 0, boxPoints[1].X);
                        boxPointsMat.Set<float>(1, 1, boxPoints[1].Y);

                        Mat boxLabelsMat = new Mat(1, 2, MatType.CV_32S);
                        boxLabelsMat.Set<int>(0, 0, boxLabels[0]);
                        boxLabelsMat.Set<int>(1, 0, boxLabels[1]);

                        // 拼接点和标签
                        pointsMat = ConcatPoints(pointsMat, boxPointsMat);
                        labelsMat = ConcatPoints(labelsMat, boxLabelsMat);
                    }

                    // 归一化坐标
                    if (normalizeCoords)
                    {
                        float scaleX = 1f / inferenceState.VideoWidth;
                        float scaleY = 1f / inferenceState.VideoHeight;
                        Cv2.Multiply(pointsMat, new Scalar(scaleX, scaleY), pointsMat);
                    }

                    // 缩放到模型输入尺寸
                    Cv2.Multiply(pointsMat, new Scalar(model.image_size, model.image_size), pointsMat);

                    // 合并新旧点
                    Mat finalPointInputs = null;
                    if (!clearOldPoints && pointInputsPerFrame.ContainsKey(frameIdx))
                    {
                        finalPointInputs = ConcatPoints(pointInputsPerFrame[frameIdx], pointsMat);
                    }
                    else
                    {
                        finalPointInputs = pointsMat;
                    }

                    // 更新输入
                    pointInputsPerFrame[frameIdx] = finalPointInputs;
                    if (maskInputsPerFrame.ContainsKey(frameIdx))
                        maskInputsPerFrame.Remove(frameIdx);

                    // 判断是否为初始条件帧
                    var objFramesTracked = (Dictionary<int, Dictionary<string, bool>>)inferenceState.FramesTrackedPerObj[objIdx];
                    bool isInitCondFrame = !objFramesTracked.ContainsKey(frameIdx);
                    bool reverse = isInitCondFrame ? false : objFramesTracked[frameIdx]["reverse"];

                    var objOutputDict = (Dictionary<string, Dictionary<int, object>>)inferenceState.OutputDictPerObj[objIdx];
                    var objTempOutputDict = (Dictionary<string, Dictionary<int, object>>)inferenceState.TempOutputDictPerObj[objIdx];
                    bool isCond = isInitCondFrame || _addAllFramesToCorrectAsCond;
                    string storageKey = isCond ? "cond_frame_outputs" : "non_cond_frame_outputs";

                    // 运行单帧推理（原_run_single_frame_inference，C#简化为接口）
                    var currentOut = _RunSingleFrameInference(
                        inferenceState, objOutputDict, frameIdx, 1, isInitCondFrame,
                        finalPointInputs, null, reverse, false, null);

                    // 更新临时输出
                    if (!objTempOutputDict[storageKey].ContainsKey(frameIdx))
                        objTempOutputDict[storageKey].Add(frameIdx, currentOut);
                    else
                        objTempOutputDict[storageKey][frameIdx] = currentOut;

                    // 合并临时输出并缩放到原视频分辨率
                    var consolidatedOut = _ConsolidateTempOutputAcrossObj(inferenceState, frameIdx, isCond, true);
                    var (_, videoResMasks) = _GetOrigVideoResOutput(inferenceState, (Mat)consolidatedOut["pred_masks_video_res"]);

                    return (frameIdx, inferenceState.ObjIds, videoResMasks);
                }

                /// <summary>
                /// 兼容旧方法，对应原add_new_points
                /// </summary>
                public (int, List<object>, Mat) AddNewPoints(params object[] args)
                {
                    // 简化：直接调用AddNewPointsOrBox
                    throw new NotImplementedException("需根据参数映射实现");
                }

                /// <summary>
                /// 添加新掩码，对应原add_new_mask方法
                /// </summary>
                public (int, List<object>, Mat) AddNewMask(
                    InferenceState inferenceState,
                    int frameIdx,
                    object objId,
                    Mat mask)
                {
                    int objIdx = _ObjIdToIdx(inferenceState, objId);
                    var pointInputsPerFrame = (Dictionary<int, Mat>)inferenceState.PointInputsPerObj[objIdx];
                    var maskInputsPerFrame = (Dictionary<int, Mat>)inferenceState.MaskInputsPerObj[objIdx];

                    // 校验掩码维度（2D）
                    if (mask.Dims != 2)
                        throw new ArgumentException("掩码必须是2维");

                    // 转换为浮点型并移到目标设备
                    Mat maskInputsOrig = new Mat();
                    mask.ConvertTo(maskInputsOrig, MatType.CV_32F);
                    maskInputsOrig = maskInputsOrig.Reshape(1, 1); // 添加batch和通道维度

                    // 缩放到模型输入尺寸
                    Mat maskInputs = new Mat();
                    if (mask.Rows != model.image_size || mask.Cols != model.image_size)
                    {
                        Cv2.Resize(maskInputsOrig, maskInputs, new Size(model.image_size, model.image_size), interpolation: InterpolationFlags.Linear);
                        Cv2.Threshold(maskInputs, maskInputs, 0.5, 1.0, ThresholdTypes.Binary);
                    }
                    else
                    {
                        maskInputs = maskInputsOrig.Clone();
                    }

                    // 更新输入
                    maskInputsPerFrame[frameIdx] = maskInputs;
                    if (pointInputsPerFrame.ContainsKey(frameIdx))
                        pointInputsPerFrame.Remove(frameIdx);

                    // 判断是否为初始条件帧
                    var objFramesTracked = (Dictionary<int, Dictionary<string, bool>>)inferenceState.FramesTrackedPerObj[objIdx];
                    bool isInitCondFrame = !objFramesTracked.ContainsKey(frameIdx);
                    bool reverse = isInitCondFrame ? false : objFramesTracked[frameIdx]["reverse"];

                    var objOutputDict = (Dictionary<string, Dictionary<int, object>>)inferenceState.OutputDictPerObj[objIdx];
                    var objTempOutputDict = (Dictionary<string, Dictionary<int, object>>)inferenceState.TempOutputDictPerObj[objIdx];
                    bool isCond = isInitCondFrame || _addAllFramesToCorrectAsCond;
                    string storageKey = isCond ? "cond_frame_outputs" : "non_cond_frame_outputs";

                    // 运行单帧推理
                    var currentOut = _RunSingleFrameInference(
                        inferenceState, objOutputDict, frameIdx, 1, isInitCondFrame,
                        null, maskInputs, reverse, false, null);

                    // 更新临时输出
                    if (!objTempOutputDict[storageKey].ContainsKey(frameIdx))
                        objTempOutputDict[storageKey].Add(frameIdx, currentOut);
                    else
                        objTempOutputDict[storageKey][frameIdx] = currentOut;

                    // 合并并缩放掩码
                    var consolidatedOut = _ConsolidateTempOutputAcrossObj(inferenceState, frameIdx, isCond, true);
                    var (_, videoResMasks) = _GetOrigVideoResOutput(inferenceState, (Mat)consolidatedOut["pred_masks_video_res"]);

                    return (frameIdx, inferenceState.ObjIds, videoResMasks);
                }

                /// <summary>
                /// 缩放到原视频分辨率，对应原_get_orig_video_res_output方法
                /// </summary>
                private (Mat, Mat) _GetOrigVideoResOutput(InferenceState inferenceState, Mat anyResMasks)
                {
                    Mat videoResMasks = new Mat();
                    if (anyResMasks.Size() == new Size(inferenceState.VideoWidth, inferenceState.VideoHeight))
                    {
                        videoResMasks = anyResMasks.Clone();
                    }
                    else
                    {
                        Cv2.Resize(anyResMasks, videoResMasks, new Size(inferenceState.VideoWidth, inferenceState.VideoHeight), interpolation: InterpolationFlags.Linear);
                    }

                    // 应用非重叠约束
                    if (_nonOverlapMasks)
                    {
                        videoResMasks = _ApplyNonOverlappingConstraints(videoResMasks);
                    }

                    return (anyResMasks, videoResMasks);
                }

        /// <summary>
        /// 合并跨对象的临时输出，对应原_consolidate_temp_output_across_obj方法
        /// </summary>
        private Dictionary<string, object> _ConsolidateTempOutputAcrossObj(
            InferenceState inferenceState,
            int frameIdx,
            bool isCond,
            bool consolidateAtVideoRes = false)
        {
            // 核心逻辑：合并所有对象的临时输出，补充缺失对象的占位符
            var consolidatedOut = new Dictionary<string, object>();
            var predMasks = new List<Mat>();

            // 遍历所有对象
            foreach (int objIdx in inferenceState.ObjIdxToId.Keys)
            {
                var objTempOutputDict = (Dictionary<string, Dictionary<int, object>>)inferenceState.TempOutputDictPerObj[objIdx];
                var storageKey = isCond ? "cond_frame_outputs" : "non_cond_frame_outputs";

                if (objTempOutputDict[storageKey].ContainsKey(frameIdx))
                {
                    // 获取当前对象的掩码
                    var objOut = (Dictionary<string, object>)objTempOutputDict[storageKey][frameIdx];
                    predMasks.Add((Mat)objOut["pred_masks"]);
                }
                else
                {
                    // 补充占位符（全0掩码）
                    Mat mat = new Mat(model.image_size, model.image_size, MatType.CV_32F);
                    predMasks.Add(mat);
                }
            }

            // 合并掩码
            Mat mergedMasks = new Mat();
            Cv2.Merge(predMasks.ToArray(), mergedMasks);
            consolidatedOut["pred_masks_video_res"] = mergedMasks;

            return consolidatedOut;
        }

                #region 未实现的核心方法（需补充）
                /// <summary>
                /// 获取图像特征，对应原_get_image_feature
                /// </summary>
                private void _GetImageFeature(InferenceState inferenceState, int frameIdx, int batchSize)
                {
                    // 需结合深度学习推理库实现（如ONNX Runtime提取特征）
                    throw new NotImplementedException();
                }

                /// <summary>
                /// 单帧推理，对应原_run_single_frame_inference
                /// </summary>
                private Dictionary<string, object> _RunSingleFrameInference(
                    InferenceState inferenceState,
                    Dictionary<string, Dictionary<int, object>> outputDict,
                    int frameIdx,
                    int batchSize,
                    bool isInitCondFrame,
                    Mat pointInputs = null,
                    Mat maskInputs = null,
                    bool reverse = false,
                    bool runMemEncoder = true,
                    Mat prevSamMaskLogits = null)
                {
                    // 需结合SAM2模型推理实现（如ONNX Runtime调用模型）
                    throw new NotImplementedException();
                }

                /// <summary>
                /// 应用非重叠约束，对应原_apply_non_overlapping_constraints
                /// </summary>
                private Mat _ApplyNonOverlappingConstraints(Mat masks)
                {
                    // 核心逻辑：确保不同对象的掩码不重叠（取每个像素最大概率的对象）
                    throw new NotImplementedException();
                }

                /// <summary>
                /// 拼接点，对应原concat_points
                /// </summary>
                private Mat ConcatPoints(Mat a, Mat b)
                {
                    if (a.Empty()) return b.Clone();
                    if (b.Empty()) return a.Clone();
                    Mat concat = new Mat();
                    Cv2.HConcat(a, b, concat);
                    return concat;
                }
                #endregion

            }
        }

