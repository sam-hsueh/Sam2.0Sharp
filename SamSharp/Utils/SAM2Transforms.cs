using OpenCvSharp;
using TorchSharp;
using TorchSharp.Modules;
using Windows.AI.MachineLearning;
using static Sam2Sharp.Utils.Classes;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Sam2Sharp.Utils
{
    public class SAM2Transforms : Module
    {
        public readonly Device device;
        public readonly ScalarType dtype;
        public (int h,int w) original_hw;
        public (float x,float y) scale=(1,1);
        //public float scaleFactor = 1.0f;
        private readonly int resolution;
        private readonly float mask_threshold;
        private readonly float max_hole_area;
        private readonly float max_sprinkle_area;
        private readonly Tensor mean1, std1, mean2, std2;
        private readonly Sequential transforms;
        public readonly float[] pixel_mean2 = [0.485f, 0.456f, 0.406f];
        public readonly float[] pixel_std12 = [0.229f, 0.224f, 0.225f];
        private readonly float[] pixel_mean1 = new float[] { 123.675f, 116.28f, 103.53f };
        private readonly float[] pixel_std1 = new float[] { 58.395f, 57.12f, 57.375f };
        //public long[] input_size = null;

        public SAM2Transforms(int resolution, Device device,ScalarType dtype, float mask_threshold, float max_hole_area = 0.0f, float max_sprinkle_area = 0.0f)
            : base("SAM2Transforms")
        {
            this.resolution = resolution;
            this.mask_threshold = mask_threshold;
            this.max_hole_area = max_hole_area;
            this.max_sprinkle_area = max_sprinkle_area;

            // 图像标准化参数
            //this.mean = tensor(pixel_mean).unsqueeze(0).unsqueeze(2).unsqueeze(3);
            //this.std = tensor(pixel_std).unsqueeze(0).unsqueeze(2).unsqueeze(3);
            this.device = device;
            this.dtype = dtype;
            mean1 = tensor(pixel_mean1).unsqueeze(-1).unsqueeze(-1).to(this.dtype, this.device);
            std1 = tensor(pixel_std1).unsqueeze(-1).unsqueeze(-1).to(this.dtype, this.device);
            mean2 = tensor(pixel_mean2).unsqueeze(-1).unsqueeze(-1).to(this.dtype, this.device);
            std2 = tensor(pixel_std12).unsqueeze(-1).unsqueeze(-1).to(this.dtype, this.device);


            //// 构建变换序列，使用自定义 Module 包装 interpolate
            //this.transforms = Sequential(
            //    new InterpolateResize(resolution),
            //    new Normalize(mean, std)
            //);

            //RegisterComponents();
        }

        //// 用于替换 Resize 的自定义 Module
        //private class InterpolateResize : Module<Tensor, Tensor>
        //{
        //    private readonly int resolution;
        //    public InterpolateResize(int resolution) : base("InterpolateResize")
        //    {
        //        this.resolution = resolution;
        //    }

        //    public override Tensor forward(Tensor x)
        //    {
        //        // 输入为 CxHxW，输出为 Cxresolutionxresolution
        //        return nn.functional.interpolate(
        //            x.unsqueeze(0),
        //            size: new[] { (long)resolution, resolution },
        //            mode: InterpolationMode.Bilinear,
        //            align_corners: false
        //        ).squeeze(0);
        //    }
        //}

        // 替换 Normalize 为标准化操作的实现
        // TorchSharp 没有直接的 Normalize Module，可以自定义一个
        private class Normalize : Module<Tensor, Tensor>
        {
            private readonly Tensor mean;
            private readonly Tensor std;

            public Normalize(Tensor mean, Tensor std) : base("Normalize")
            {
                this.mean = mean;
                this.std = std;
            }

            public override Tensor forward(Tensor x)
            {
                // 标准化: (x - mean) / std
                return (x - mean) / std;
            }
        }

        //public Tensor forward(Tensor x)
        //{
        //    // 转换为张量并应用变换
        //    var tensor = ToTensor(x);
        //    return transforms.forward(tensor);
        //}

        public (List<Tensor>,List<(int h,int w)>) forward_batch(List<string> imgList)
        {
            List<(Tensor imgTenor,(int h,int w))> imgBatch = imgList.Select((img) => this.mat_tensor(img)).ToList();
            var imgTensors = imgBatch.Select(t => t.Item1.squeeze(0)).ToList();
            var orgSize = imgBatch.Select(t => (t.Item2)).ToList();
            return (imgTensors, orgSize);
        }
        int cat = 1;
        public (Tensor, (int H, int W)) mat_tensor(string img_path)
        {
            using var _ = no_grad();
            long start = DateTime.Now.Ticks;
            (Tensor x, int H, int W) = Sam2Sharp.Tools.ImageTools._load_img_as_tensor(img_path, resolution);
            original_hw = (H, W);
            scale = (((float)resolution / W), ((float)resolution / H));
            Tensor mean, std;
            if (cat == 0)
            {
                mean = mean1;
                std = std1;
            }
            else if(cat == 1)
            {
                x /= 255.0f;
                mean = mean2;
                std = std2;
            }
            else if(cat == 2)
            {
                var image = Cv2.ImRead(img_path, ImreadModes.Color);
                x = Tools.ImageTools.GetTensorFromImage(image);
                long w = x.shape[2];
                long h = x.shape[1];
                original_hw = ((int)h, (int)w);
                x = x.to(ScalarType.Float32) / 255.0f;
                x = x.to(dtype, device);
                x = x.unsqueeze(0);
                scale = (1f, 1f);

                x = ((x - mean2) / std2);//No use mean
                long padh = this.resolution - h;
                long padw = this.resolution - w;
                x = torch.nn.functional.pad(x, new long[] { 0, padw, 0, padh });
                return (x.MoveToOuterDisposeScope(),(H,W));
            }
            else
            {
                var image = Cv2.ImRead(img_path, ImreadModes.Color);
                x = Tools.ImageTools.GetTensorFromImage(image);
                long w = x.shape[2];
                long h = x.shape[1];
                original_hw = ((int)h, (int)w);
                x = x.to(dtype, device);
                x = x.unsqueeze(0);
                scale = (1f, 1f);

                x = ((x - mean1) / std1);//No use mean
                long padh = this.resolution - h;
                long padw = this.resolution - w;
                x = torch.nn.functional.pad(x, new long[] { 0, padw, 0, padh });
                return (x.MoveToOuterDisposeScope(), (H, W));
            }
            x = x.to(dtype, device);
            x = x.unsqueeze(0);
            x = ((x - mean) / std);//No use mean
            return (x.MoveToOuterDisposeScope(), (H, W));
        }
        public Tensor mat_tensor(Mat image)
        {
            using var _ = no_grad();
            long start = DateTime.Now.Ticks;
            //Mat rgbMat = new Mat();
            //Cv2.CvtColor(image, rgbMat, ColorConversionCodes.BGR2RGB); // 对应convert("RGB")
            Mat resizedMat = new Mat();
            if(resolution!=image.Width||resolution!=image.Height)
                Cv2.Resize(image, resizedMat, new OpenCvSharp.Size(resolution, resolution)); // 对应resize
            else
                resizedMat = image; 
            // 4. 数据类型校验（对应原img_np.dtype == np.uint8）
            if (resizedMat.Type() != MatType.CV_8UC3)
            {
                throw new Exception($"Unknown image dtype: {resizedMat.Type()}");
            }
            Tensor x = torchvision.io.read_image(resizedMat.ToMemoryStream(), torchvision.io.ImageReadMode.RGB);
            var H = image.Height;
            var W = image.Width;

            resizedMat.Release();
            original_hw = (H, W);
            scale = (((float)resolution / W), ((float)resolution / H));
            Tensor mean, std;
            if (cat == 0)
            {
                mean = mean1;
                std = std1;
            }
            else/* if (cat == 1)*/
            {
                x /= 255.0f;
                mean = mean2;
                std = std2;
            }
            //else if (cat == 2)
            //{
            //    var image = Cv2.ImRead(img_path, ImreadModes.Color);
            //    x = Tools.ImageTools.GetTensorFromImage(image);
            //    long w = x.shape[2];
            //    long h = x.shape[1];
            //    original_hw = ((int)h, (int)w);
            //    x = x.to(ScalarType.Float32) / 255.0f;
            //    x = x.to(dtype, device);
            //    x = x.unsqueeze(0);
            //    scale = (1f, 1f);

            //    x = ((x - mean2) / std2);//No use mean
            //    long padh = this.resolution - h;
            //    long padw = this.resolution - w;
            //    x = torch.nn.functional.pad(x, new long[] { 0, padw, 0, padh });
            //    return (x.MoveToOuterDisposeScope(), (H, W));
            //}
            //else
            //{
            //    var image = Cv2.ImRead(img_path, ImreadModes.Color);
            //    x = Tools.ImageTools.GetTensorFromImage(image);
            //    long w = x.shape[2];
            //    long h = x.shape[1];
            //    original_hw = ((int)h, (int)w);
            //    x = x.to(dtype, device);
            //    x = x.unsqueeze(0);
            //    scale = (1f, 1f);

            //    x = ((x - mean1) / std1);//No use mean
            //    long padh = this.resolution - h;
            //    long padw = this.resolution - w;
            //    x = torch.nn.functional.pad(x, new long[] { 0, padw, 0, padh });
            //    return (x.MoveToOuterDisposeScope(), (H, W));
            //}
            x = x.to(dtype, device);
            x = x.unsqueeze(0);
            x = ((x - mean) / std);//No use mean
            return x.MoveToOuterDisposeScope();
            //return (x.MoveToOuterDisposeScope(), (H, W));
        }
        public (Tensor,int h,int w) mat_tensor2(string img_path)
        {
            using var _ = no_grad();
            var image = Cv2.ImRead(img_path, ImreadModes.Color);
            long start = DateTime.Now.Ticks;
            original_hw =(image.Height, image.Width);
            scale = (((float)resolution/original_hw.w),((float)resolution/original_hw.h));
            var targetsize = new OpenCvSharp.Size(resolution, resolution);
            var nimg = new Mat();
            Cv2.Resize(image, nimg, targetsize);
            Tensor x = Sam2Sharp.Tools.ImageTools.GetTensorFromImage(nimg);
            x = x.to(ScalarType.Float32) / 255.0f;
            x = x.to(dtype, device);
            //var imgTensor2 = torchvision.transforms.functional.resize(imgTensor,resolution, resolution);

            x = x.unsqueeze(0);
            x = ((x - mean2) / std2);//No use mean
            return (x,image.Height, image.Width);
        }
        public Tensor mat_tensor1(string img_path)
        {
            using var _ = no_grad();
            var image = Cv2.ImRead(img_path, ImreadModes.Color);
            long start = DateTime.Now.Ticks;
            Tensor x = Tools.ImageTools.GetTensorFromImage(image);
            long w = x.shape[2];
            long h = x.shape[1];
            original_hw = ((int)h, (int)w);
            x = x.to(ScalarType.Float32)/255.0f;
            x = x.to(dtype, device);
            x = x.unsqueeze(0);
            scale = (1f, 1f);

            //      original_hw = new long[] { image.Width, image.Height };
            //Tensor pixel_mean_tensor = tensor(this.pixel_mean).unsqueeze(-1).unsqueeze(-1).to(x.dtype, x.device);
            //Tensor pixel_std_tensor = tensor(this.pixel_std).unsqueeze(-1).unsqueeze(-1).to(x.dtype, x.device);
            x = ((x - mean2) / std2);//No use mean
                                  // Pad
            //long nh = x.shape[2];
            //long nw = x.shape[3];
            long padh = this.resolution - h;
            long padw = this.resolution - w;
            x = torch.nn.functional.pad(x, new long[] { 0, padw, 0, padh });
            //var dd = x.to(ScalarType.Float32);
            //var array = dd.data<float>().ToArray();
            cat = 2;
            return x.MoveToOuterDisposeScope();
        }
        //public Tensor mat_tensor0(Mat image)
        //{
        //    using var _ = no_grad();
        //    long start = DateTime.Now.Ticks;
        //    Tensor x = Tools.ImageTools.GetTensorFromImage(image);
        //    long w = x.shape[2];
        //    long h = x.shape[1];
        //    scaleFactor = Math.Min((float)resolution / w, (float)resolution / h);
        //    scaleFactor = 1.0f;
        //    int newW = (int)Math.Ceiling(w * scaleFactor);
        //    int newH = (int)Math.Ceiling(h * scaleFactor);
        //    x = torchvision.transforms.functional.resize(x, newH, newW);
        //    original_hw = new long[] { h, w };
        //    x = x.unsqueeze(0);

        //    //      original_hw = new long[] { image.Width, image.Height };
        //    Tensor pixel_mean_tensor = tensor(this.pixel_mean).unsqueeze(-1).unsqueeze(-1).to(x.dtype, x.device);
        //    Tensor pixel_std_tensor = tensor(this.pixel_std).unsqueeze(-1).unsqueeze(-1).to(x.dtype, x.device);
        //    x = ((x - std) / std);//No use mean
        //                          // Pad
        //    input_size = [(long)newH, (long)newW];
        //    //long nh = x.shape[2];
        //    //long nw = x.shape[3];
        //    long padh = this.resolution - newH;
        //    long padw = this.resolution - newW;
        //    x = torch.nn.functional.pad(x, new long[] { 0, padw, 0, padh });
        //    //var dd = x.to(ScalarType.Float32);
        //    //var array = dd.data<float>().ToArray();
        //    return x.MoveToOuterDisposeScope();
        //}


        public (Tensor, Tensor, Tensor, Tensor) points_to_Tensor(List<SamPoint> points, Tensor box, Tensor maskLogits, bool normalizeCoords=true, (int h, int w)? orig_hw=null)
        {
            using var _ = no_grad();
            //using var __ = NewDisposeScope();
            Tensor pointsTensor = null;
            Tensor labelsTensor = null;

            if (points is not null)
            {
                if (points.Count > 0)
                {
                    pointsTensor = torch.zeros(new long[] { 1, points?.Count ?? 0, 2 });
                    labelsTensor = torch.zeros(new long[] { 1, points?.Count ?? 0 });
                    for (int i = 0; i < (points?.Count ?? 0); i++)
                    {
                        Classes.SamPoint point = points![i];
                        //var rw = resolution / orig_hw!.Value.w;
                        //var rh = resolution / orig_hw!.Value.h;
                        pointsTensor[0, i, 0] = (int)(point.X *scale.x);
                        pointsTensor[0, i, 1] = (int)(point.Y *scale.y);
                        labelsTensor[0, i] = point.Label ?? true ? 1 : 0;
                    }
                }
            }
            return (pointsTensor, labelsTensor,null,null);
        }

        public Tensor transform_coords(Tensor coords, bool normalize = false, (int h, int w)? orig_hw = null)
        {
            if (normalize)
            {
                if (orig_hw == null)
                    throw new ArgumentNullException(nameof(orig_hw), "Original height and width must be provided when normalizing coordinates");

                var (h, w) = orig_hw.Value;
                var coordsClone = coords.clone();
                coordsClone[.., TensorIndex.Ellipsis, 0] = coordsClone[.., TensorIndex.Ellipsis, 0] / w;
                coordsClone[.., TensorIndex.Ellipsis, 1] = coordsClone[.., TensorIndex.Ellipsis, 1] / h;
                coords = coordsClone;
            }

            return coords;
        }

        public Tensor transform_boxes(Tensor boxes, bool normalize = false, (int h, int w)? orig_hw = null)
        {
            var reshaped = boxes.reshape(-1, 2, 2);
            var transformed = transform_coords(reshaped, normalize, orig_hw);
            return transformed;
        }

        public Tensor postprocess_masks(Tensor masks, (int h, int w) orig_hw)
        {
            //无缩放
            if (cat >= 2)
            {
                masks = torch.nn.functional.interpolate(masks, new long[] { this.resolution, this.resolution }, mode: InterpolationMode.Bilinear, align_corners: false);
                masks = masks[TensorIndex.Ellipsis, ..(int)(orig_hw.h), ..(int)(orig_hw.w)];
                masks = torch.nn.functional.interpolate(masks, [(long)orig_hw.h, (long)orig_hw.w], mode: InterpolationMode.Bilinear, align_corners: false);
                return masks;
            }
            else
            {
                return torch.nn.functional.interpolate(masks,
                                size: [orig_hw.h, orig_hw.w],
                                mode: InterpolationMode.Bilinear,
                                align_corners: false);
            }



                var inputMasks = masks;
            masks = masks.to_type(ScalarType.Float32); // 修正此处
            var maskFlat = masks.flatten(0, 1).unsqueeze(1);

            try
            {
                if (max_hole_area > 0)
                {
                    var (labels, areas) = get_connected_components(maskFlat <= mask_threshold);
                    var isHole = (labels > 0) & (areas <= max_hole_area);
                    isHole = isHole.reshape(masks.shape);
                    masks = torch.where(isHole, tensor(mask_threshold + 10.0f), masks);
                }

                if (max_sprinkle_area > 0)
                {
                    var (labels, areas) = get_connected_components(maskFlat > mask_threshold);
                    var isHole = (labels > 0) & (areas <= max_sprinkle_area);
                    isHole = isHole.reshape(masks.shape);
                    masks = torch.where(isHole, tensor(mask_threshold - 10.0f), masks);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: {ex.Message}\nSkipping post-processing step...");
                masks = inputMasks;
            }

            // 调整掩码到原始图像尺寸
            masks = nn.functional.interpolate(
                masks,
                size: new[] { (long)orig_hw.h, orig_hw.w },
                mode: InterpolationMode.Bilinear,
                align_corners: false
            );

            return masks;
        }

        private Tensor ToTensor(Tensor x)
        {
            // 假设输入是HxWxC格式的图像，转换为CxHxW并归一化到[0,1]
            return x.permute(2, 0, 1).to(ScalarType.Float32) / 255.0f;
        }

        // 注意：需要实现或引入连通组件分析的功能
        private (Tensor labels, Tensor areas) get_connected_components(Tensor mask)//联通域检查
        {
            // 这里需要实现连通组件分析
            // 实际应用中可能需要使用TorchSharp的扩展或自定义实现
            throw new NotImplementedException("Connected components analysis needs implementation");
        }
    }
}
