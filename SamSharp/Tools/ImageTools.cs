using OpenCvSharp;
using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace Sam2Sharp.Tools
{
	internal class ImageTools
	{
        public static (Tensor img, int video_height, int video_width) _load_img_as_tensor(string img_path, int image_size)
        {
            // 1. 检查文件是否存在
            if (!System.IO.File.Exists(img_path))
            {
                throw new FileNotFoundException("图片文件不存在", img_path);
            }

            // 2. 使用OpenCvSharp加载图片（替代PIL.Image.open + np.array）
            using (var imgMat = Cv2.ImRead(img_path, ImreadModes.Color))
            {
                if (imgMat.Empty())
                {
                    throw new Exception($"无法加载图片：{img_path}");
                }

                // 获取原始图片尺寸（对应原img_pil.size）
                int video_width = imgMat.Cols;
                int video_height = imgMat.Rows;

                // 3. 转换为RGB格式（OpenCV默认BGR，需转换）+ 调整尺寸
                Mat rgbMat = new Mat();
                Cv2.CvtColor(imgMat, rgbMat, ColorConversionCodes.BGR2RGB); // 对应convert("RGB")
                Mat resizedMat = new Mat();
                Cv2.Resize(rgbMat, resizedMat, new OpenCvSharp.Size(image_size, image_size)); // 对应resize

                // 4. 数据类型校验（对应原img_np.dtype == np.uint8）
                if (resizedMat.Type() != MatType.CV_8UC3)
                {
                    throw new Exception($"Unknown image dtype: {resizedMat.Type()} on {img_path}");
                }

                //// 5. 归一化（uint8 → float32，除以255.0）
                //Mat floatMat = new Mat();
                //resizedMat.ConvertTo(floatMat, MatType.CV_32FC3, 1.0 / (norm ? 255.0 : 1.0f)); // 直接转换并归一化

                //// 6. 将OpenCvSharp Mat转换为TorchSharp Tensor
                //// 步骤1：Mat → float数组（HWC格式）
                //float[] floatData = new float[floatMat.Total() * floatMat.Channels()];
                //floatMat.GetArray<float>(out floatData);

                //// 步骤2：构造HWC格式的Tensor
                //Tensor imgTensor = torch.tensor(floatData, ScalarType.Float32).reshape(new long[] { image_size, image_size, 3 });
                Tensor x = torchvision.io.read_image(resizedMat.ToMemoryStream(), torchvision.io.ImageReadMode.RGB);
                // 步骤3：维度置换 HWC → CHW（对应permute(2,0,1)）
                //       x = x.permute(2, 0, 1);

                // 释放临时Mat资源
                rgbMat.Release();
                resizedMat.Release();
              //  floatMat.Release();
                return (x, video_height, video_width);
            }
        }
        internal static Tensor GetTensorFromImage(SKBitmap skBitmap)
		{
			using (MemoryStream stream = new MemoryStream())
			{
				skBitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
				stream.Position = 0;
				Tensor tensor = torchvision.io.read_image(stream, torchvision.io.ImageReadMode.RGB);
				return tensor;
			}
		}
        internal static Tensor GetTensorFromImage(Mat mat, torchvision.io.ImageReadMode readMode = torchvision.io.ImageReadMode.RGB)
        {
            Tensor tensor = torchvision.io.read_image(mat.ToMemoryStream(), readMode);
            return tensor;
        }

        internal static SKBitmap GetImageFromTensor(Tensor tensor)
		{
			using (MemoryStream memoryStream = new MemoryStream())
			{
				torchvision.io.write_png(tensor.cpu(), memoryStream);
				memoryStream.Position = 0;
				SKBitmap skBitmap = SKBitmap.Decode(memoryStream);
				return skBitmap;
			}
		}
	}
}
