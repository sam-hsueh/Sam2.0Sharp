using OpenCvSharp;
using Sam2Sharp.Modeling;
using SkiaSharp;
using TorchSharp;
using static Sam2Sharp.Utils.Classes;
using static TorchSharp.torch;

namespace Sam2Sharp.Utils
{
	/// <summary>
	///  Predict masks for the given input prompts, using the currently set image. Input prompts are batched torch tensors and are expected to already be transformed to the input frame using ResizeLongestSide.
	/// </summary>
	/// <param name="point_coords">A BxNx2 array of point prompts to the model.Each point is in (X, Y) in pixels.</param>
	/// <param name="point_labels">A BxN array of labels for the point prompts. 1 indicates a foreground point and 0 indicates a background point.</param>
	/// <param name="mask_input">A low resolution mask input to the model, typically coming from a previous prediction iteration.Has form Bx1xHxW, where for SAM, H= W = 256.Masks returned by a previous iteration of the predict method do not need further transformation.</param>
	/// <param name="multimask_output">If true, the model will return three masks. For ambiguous input prompts(such as a single click), this will often produce better masks than a single prediction.If only a single mask is needed, the model's predicted quality score can be used to select the best mask.For non-ambiguous prompts, such as multiple input prompts, multimask_output=False can give better results.</param>
	/// <param name="return_logits">If true, returns un-thresholded masks logits instead of a binary mask.</param>
	/// <returns>(torch.Tensor): The output masks in BxCxHxW format, where C is the number of masks, and(H, W) is the original image size.<br/>
	/// (torch.Tensor): An array of shape BxC containing the model's predictions for the quality of each mask.<br/>
	/// (torch.Tensor): An array of shape BxCxHxW, where C is the number of masks and H = W = 256.These low res logits can be passed to	a subsequent iteration as mask input.</returns>

	public class Sam2Predictor
	{
		private readonly Sam2Base model;
		private readonly Device device;
		private readonly ScalarType dtype;

		private long[] original_size = null;
		private float scaleFactor = 0.0f;


		/// <summary>
		/// Init Predictor, you don't have to choose Vit-b, Vit-l or Vit-H. It will be auto selected when loading model.
		/// </summary>
		/// <param name="checkpointPath">Checkpoint Path</param>
		/// <param name="device">Sam2 Device, it's CPU or Cuda.</param>
		public Sam2Predictor(string checkpointPath, Sam2Device device = Sam2Device.CPU, Sam2ScalarType dtype = Sam2ScalarType.Float32)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager(100);
			this.device = new Device((DeviceType)device);
			this.dtype = (ScalarType)dtype;
			this.model = BuildSam2.BuildSam2Model(checkpointPath, this.device, this.dtype);
		}

		public void SetImage(SKBitmap image, int maxImageSize = 1024)
		{
            using var _ = no_grad();
            long start = DateTime.Now.Ticks;
            Tensor imgTensor = Tools.ImageTools.GetTensorFromImage(image);
			long w = imgTensor.shape[2];
			long h = imgTensor.shape[1];
			scaleFactor = Math.Min((float)maxImageSize / w, (float)maxImageSize / h);
			int newW = (int)Math.Ceiling(w * scaleFactor);
			int newH = (int)Math.Ceiling(h * scaleFactor);
			imgTensor = torchvision.transforms.functional.resize(imgTensor, newH, newW);
			original_size = new long[] { h, w };
			model.SetImage(imgTensor,this.device,this.dtype);
            long mid = DateTime.Now.Ticks;
            long MidGIelapsedMs = (mid - start) / TimeSpan.TicksPerMillisecond;
		}
        public void SetImage(Mat image, int maxImageSize = 1024)
        {
            using var _ = no_grad();
            long start = DateTime.Now.Ticks;
            Tensor imgTensor = Tools.ImageTools.GetTensorFromImage(image);
            long w = imgTensor.shape[2];
            long h = imgTensor.shape[1];
            scaleFactor = Math.Min((float)maxImageSize / w, (float)maxImageSize / h);
            int newW = (int)Math.Ceiling(w * scaleFactor);
            int newH = (int)Math.Ceiling(h * scaleFactor);
            imgTensor = torchvision.transforms.functional.resize(imgTensor, newH, newW);
            original_size = new long[] { h, w };
            long mid = DateTime.Now.Ticks;
            long MidGIelapsedMs = (mid - start) / TimeSpan.TicksPerMillisecond;
            model.SetImage(imgTensor, this.device, this.dtype);
        }

        public List<PredictOutput> Predict(List<Sam2Point> points = null)
		{
			using var _ = no_grad();
			using var __ = NewDisposeScope();
			model.eval();

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
						Sam2Point point = points[i];
						pointsTensor[0, i, 0] = point.X * scaleFactor;
						pointsTensor[0, i, 1] = point.Y * scaleFactor;
						labelsTensor[0, i] = point.Label??true ? 1 : 0;
					}
				}
			}

			BatchedInput batchedInput = new BatchedInput
			{
				Point_coords = pointsTensor,
				Point_labels = labelsTensor,
				Original_size = original_size,
				Input_size = new long[] { (long)(original_size[0] * scaleFactor), (long)(original_size[1] * scaleFactor) },
			};

			BatchedOutput output = model.forward(batchedInput, false);
			List<PredictOutput> predictOutputs = new List<PredictOutput>();

			for (int i = 0; i < output.Masks.shape[0]; i++)
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
			output.Dispose();
			GC.Collect();
			return predictOutputs;
		}
	}
}
