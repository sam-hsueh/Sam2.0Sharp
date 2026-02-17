using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Sam2Sharp.Modeling
{
	public class Conv2dBN : Module<Tensor, Tensor>
	{
		private readonly Conv2d c;
		private readonly BatchNorm2d bn;
		private readonly long groups;
		private readonly int stride;
		private readonly int padding;
		private readonly int dilation;
		public Conv2dBN(long inputChannels, long outputChannels, int kernel_size = 1, int stride = 1, int padding = 0, int dilation = 1, long groups = 1, float bnWeightInit = 1.0f, string name = "Conv2dBN") : base(name)
		{
			this.groups = groups;
			this.stride = stride;
			this.padding = padding;
			this.dilation = dilation;
			c = Conv2d(inputChannels, outputChannels, kernel_size: kernel_size, stride: stride, padding: padding, dilation: dilation, groups: groups, bias: false);
			bn = BatchNorm2d(outputChannels);
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			using (NewDisposeScope())
			{
				if (this.training)
				{
					return bn.forward(c.forward(input)).MoveToOuterDisposeScope();
				}
				else
				{
					using (var _ = torch.no_grad())
					{
						(Device device, ScalarType dtype) = Common.GetDeviceAndScaleType(this);
						Tensor cWeights = c.weight;
						Tensor bnWeights = bn.weight;
						Tensor bnBiases = bn.bias;
						Tensor bnRunningVar = bn.running_var;
						Tensor bnEps = torch.tensor(1e-5); //应该是bn.eps
						Tensor bnRunningMean = bn.running_mean;

						// Compute the fused weights and biases
						Tensor w = bnWeights / (bnRunningVar + bnEps).sqrt();
						w = cWeights * w.unsqueeze(1).unsqueeze(1).unsqueeze(1);
						Tensor b = bnBiases - bnRunningMean * bnWeights / (bnRunningVar + bnEps).sqrt();

						// Create a new Conv2d module with the fused weights
						Conv2d fusedModule = Conv2d(w.size(1) * groups, w.size(0), w.shape[2..][0], stride: stride, padding: padding, dilation: dilation, groups: groups, device: device, dtype: dtype);
						fusedModule.weight.copy_(w);
						fusedModule.bias.copy_(b);

						return fusedModule.forward(input).MoveToOuterDisposeScope();
					}
				}
			}
		}
	}
	public class PatchEmbed : Module<Tensor, Tensor>
	{
		private readonly Sequential seq;
		public long NumPatches { get; private set; }
		public long[] PatchesResolution { get; private set; }
		public long PatchesResolution1 { get; private set; }

		public PatchEmbed(long inChans, long embed_dim, int resolution, Func<Module<Tensor, Tensor>> activation, string name = "patch_embed") : base(name)
		{
			var imgSize = new long[] { resolution, resolution };
			PatchesResolution = new long[] { imgSize[0] / 4, imgSize[1] / 4 };
			NumPatches = PatchesResolution[0] * PatchesResolution[1];

			seq = Sequential(
				new Conv2dBN(inChans, embed_dim / 2, 3, 2, 1),
				activation(),
				new Conv2dBN(embed_dim / 2, embed_dim, 3, 2, 1)
			);
			RegisterComponents();
		}

		public override Tensor forward(Tensor x)
		{
			return seq.forward(x);
		}
	}
	public class DropPath : Module<Tensor, Tensor>
	{
		private float drop_prob;
		private bool scale_by_keep;

		public DropPath(float drop_prob = 0.0f, bool scaleByKeep = true) : base("DropPath")
		{
			this.drop_prob = drop_prob;
			scale_by_keep = scaleByKeep;
			RegisterComponents();
		}

		public override Tensor forward(Tensor x)
		{
			using (NewDisposeScope())
			{
				if (drop_prob == 0.0 || !training)
				{
					return x;
				}

				var keep_prob = 1 - drop_prob;
				var shape = new long[] { x.shape[0] }.Concat(Enumerable.Repeat(1L, (int)x.ndim - 1)).ToArray();
				var random_tensor = x.new_empty(shape).bernoulli_(keep_prob);

				if (keep_prob > 0.0 && scale_by_keep)
				{
					random_tensor.div_(keep_prob);
				}
				return (x * random_tensor).MoveToOuterDisposeScope();
			}
		}
	}

	public class MBConv : Module<Tensor, Tensor>
	{
		private Conv2dBN conv1;
		private Module<Tensor, Tensor> act1;
		private Conv2dBN conv2;
		private Module<Tensor, Tensor> act2;
		private Conv2dBN conv3;
		private Module<Tensor, Tensor> act3;
		private Module<Tensor, Tensor> drop_path;

		public MBConv(long inChans, long outChans, float expandRatio, Func<Module<Tensor, Tensor>> activation, float dropPath, string name = "MBConv") : base(name)
		{
			long hiddenChans = (long)(inChans * expandRatio);

			conv1 = new Conv2dBN(inChans, hiddenChans, 1);
			act1 = activation();

			conv2 = new Conv2dBN(hiddenChans, hiddenChans, 3, stride: 1, padding: 1, groups: hiddenChans);
			act2 = activation();

			conv3 = new Conv2dBN(hiddenChans, outChans, 1, bnWeightInit: 0.0f);
			act3 = activation();

			drop_path = dropPath > 0 ? new DropPath(dropPath) : Identity();

			RegisterComponents();
		}

		public override Tensor forward(Tensor x)
		{
			using (NewDisposeScope())
			{
				Tensor shortcut = x;

				x = conv1.forward(x);
				x = act1.forward(x);

				x = conv2.forward(x);
				x = act2.forward(x);

				x = conv3.forward(x);
				x = drop_path.forward(x);

				x += shortcut;
				x = act3.forward(x);

				return x.MoveToOuterDisposeScope();
			}
		}
	}

	public class PatchMerging : Module<Tensor, Tensor>
	{
		private readonly Module<Tensor, Tensor> act;
		private readonly Conv2dBN conv1;
		private readonly Conv2dBN conv2;
		private readonly Conv2dBN conv3;
		private readonly Tuple<int, int> inputResolution;
		private readonly long dim;
		private readonly long out_dim;

		public PatchMerging(Tuple<int, int> inputResolution, long dim, long out_dim, Func<Module<Tensor, Tensor>> activation, string name = "PatchMerging") : base(name)
		{
			this.inputResolution = inputResolution;
			this.dim = dim;
			this.out_dim = out_dim;
			this.act = activation();

			conv1 = new Conv2dBN(dim, out_dim, 1, 1, 0);
			int strideC = out_dim == 320 || out_dim == 448 || out_dim == 576 ? 1 : 2;
			conv2 = new Conv2dBN(out_dim, out_dim, 3, strideC, 1, groups: out_dim);
			conv3 = new Conv2dBN(out_dim, out_dim, 1, 1, 0);

			RegisterComponents();
		}

		public override Tensor forward(Tensor x)
		{
			if (x.ndim == 3)
			{
				var (H, W) = inputResolution;
				var B = x.shape[0];
				// (B, C, H, W)
				x = x.view(B, H, W, -1).permute(0, 3, 1, 2);
			}

			x = conv1.forward(x);
			x = act.forward(x);
			x = conv2.forward(x);
			x = act.forward(x);
			x = conv3.forward(x);
			x = x.flatten(2).transpose(1, 2);
			return x;
		}
	}

	public class ConvLayer : Module<Tensor, Tensor>
	{
		private readonly ModuleList<MBConv> blocks;
		private readonly Module<Tensor, Tensor> downsample;
		private readonly bool useCheckpoint;

		public ConvLayer(long dim, Tuple<int, int> inputResolution, int depth, Func<Module<Tensor, Tensor>> activation, List<float> dropPath = null, Func<Tuple<int, int>, long, long, Func<Module<Tensor, Tensor>>, Module<Tensor, Tensor>> downsample = null, bool useCheckpoint = false, long? out_dim = null, float convExpandRatio = 4.0f, string name = "ConvLayer") : base(name)
		{
			this.useCheckpoint = useCheckpoint;
			if (dropPath is null)
			{
				dropPath = new List<float>() { 0 };
			}
			MBConv[] blockList = new MBConv[depth];

			for (int i = 0; i < depth; i++)
			{
				float dp = dropPath[0];
				if (dropPath.Count > 1)
				{
					dp = dropPath[i];
				}
				blockList[i] = new MBConv(dim, dim, convExpandRatio, activation, dp);
			}

			blocks = new ModuleList<MBConv>(blockList);

			if (downsample is not null)
			{
				this.downsample = downsample(inputResolution, dim, out_dim.Value, activation);
			}
			RegisterComponents();
		}

		public override Tensor forward(Tensor x)
		{
			foreach (var blk in blocks)
			{
				x = blk.forward(x);
			}

			if (downsample is not null)
			{
				x = downsample.forward(x);
			}

			return x;
		}
	}

	public class Mlp : Module<Tensor, Tensor>
	{
		private readonly LayerNorm norm;
		private readonly Linear fc1;
		private readonly Linear fc2;
		private readonly Module<Tensor, Tensor> act;
		private readonly Dropout drop;

		public Mlp(long inFeatures, long? hiddenFeatures = null, long? outFeatures = null, Func<Module<Tensor, Tensor>> actLayer = null, float drop = 0.0f, string name = "Mlp") : base(name)
		{
			actLayer = actLayer ?? GELU;
			outFeatures = outFeatures ?? inFeatures;
			hiddenFeatures = hiddenFeatures ?? inFeatures;

			norm = LayerNorm(inFeatures);
			fc1 = Linear(inFeatures, hiddenFeatures.Value);
			fc2 = Linear(hiddenFeatures.Value, outFeatures.Value);
			act = actLayer();
			this.drop = Dropout(drop);
			RegisterComponents();
		}

		public override Tensor forward(Tensor x)
		{
			x = norm.forward(x);
			x = act.forward(fc1.forward(x));
			x = drop.forward(x);
			x = drop.forward(fc2.forward(x));
			return x;
		}
	}

	public class Attention : Module<Tensor, Tensor>
	{
		private readonly int num_heads;
		private readonly float scale;
		private readonly long keyDim;
		private readonly long nhKeyDim;
		private readonly int d;
		private readonly int dh;
		private readonly float attnRatio;
		private readonly Module<Tensor, Tensor> norm;
		private readonly Module<Tensor, Tensor> qkv;
		private readonly Module<Tensor, Tensor> proj;
		private readonly Tensor attention_biases;
		private readonly Tensor attention_bias_idxs;
		//private Tensor ab;

		public Attention(long dim, long keyDim, int num_heads = 8, float attnRatio = 4, Tuple<int, int> resolution = null, string name = "Attention") : base(name)
		{
			if (resolution == null)
			{
				resolution = Tuple.Create(14, 14);
			}
			this.num_heads = num_heads;
			this.scale = (float)Math.Pow(keyDim, -0.5);
			this.keyDim = keyDim;
			this.nhKeyDim = keyDim * num_heads;
			this.d = (int)(attnRatio * keyDim);
			this.dh = (int)(attnRatio * keyDim) * num_heads;
			this.attnRatio = attnRatio;
			var h = this.dh + nhKeyDim * 2;

			var points = Enumerable.Range(0, resolution.Item1)
				.SelectMany(x => Enumerable.Range(0, resolution.Item2), (x, y) => new { X = x, Y = y })
				.ToList();
			int N = points.Count;
			var attentionOffsets = new Dictionary<(int, int), int>();
			var idxs = new List<int>();
			foreach (var p1 in points)
			{
				foreach (var p2 in points)
				{
					var offset = (Math.Abs(p1.X - p2.X), Math.Abs(p1.Y - p2.Y));
					if (!attentionOffsets.ContainsKey(offset))
					{
						attentionOffsets[offset] = attentionOffsets.Count;
					}
					idxs.Add(attentionOffsets[offset]);
				}
			}

			this.attention_biases = Parameter(zeros(num_heads, attentionOffsets.Count));
			attention_bias_idxs = LongTensor(idxs.ToArray()).view(N, N);
			this.register_buffer("attention_bias_idxs", attention_bias_idxs, persistent: false);

			norm = LayerNorm(dim);
			qkv = Linear(dim, h);
			proj = Linear(dh, dim);
			RegisterComponents();
		}

		public override Tensor forward(Tensor x)
		{
			using (NewDisposeScope())
			{
				var shape = x.shape;
				var (B, N) = (shape[0], shape[1]);

				// Normalization
				x = norm.forward(x);

				var qkv = this.qkv.forward(x);
				// (B, N, num_heads, d)
				Tensor[] splitTensors = qkv.view(B, N, num_heads, -1).split(new long[] { keyDim, keyDim, d }, dim: 3);
				var (q, k, v) = (splitTensors[0], splitTensors[1], splitTensors[2]);
				// (B, num_heads, N, d)
				q = q.permute(0, 2, 1, 3);
				k = k.permute(0, 2, 1, 3);
				v = v.permute(0, 2, 1, 3);
				var t = attention_biases[TensorIndex.Slice(), TensorIndex.Tensor(attention_bias_idxs)];

				var attn = q.matmul(k.transpose(-2, -1)) * scale + t;
				attn = softmax(attn, dim: -1);
				x = attn.matmul(v).transpose(1, 2).reshape(B, N, dh);
				x = proj.forward(x);
				return x.MoveToOuterDisposeScope();
			}
		}
	}

	public class TinyViTBlock : Module<Tensor, Tensor>
	{
		private readonly long dim;
		private readonly Tuple<int, int> inputResolution;
		private readonly int num_heads;
		private readonly int window_size;
		private readonly float mlp_ratio;
		private readonly float drop;
		private readonly float dropPath;
		private readonly int localConvSize;
		private readonly Module<Tensor, Tensor> activation;
		private readonly Attention attn;
		private readonly Mlp mlp;
		private readonly Conv2dBN local_conv;
		private readonly Module<Tensor, Tensor> dropPathLayer;

		public TinyViTBlock(long dim, Tuple<int, int> inputResolution, int num_heads, int window_size = 7, float mlp_ratio = 4, float drop = 0, float dropPath = 0, int localConvSize = 3, Func<Module<Tensor, Tensor>> activation = null, string name = "TinyViTBlock") : base(name)
		{
			this.dim = dim;
			this.inputResolution = inputResolution;
			this.num_heads = num_heads;
			this.window_size = window_size;
			this.mlp_ratio = mlp_ratio;
			this.drop = drop;
			this.dropPath = dropPath;
			this.localConvSize = localConvSize;
			this.activation = activation == null ? nn.GELU() : activation();

			this.attn = new Attention(dim, dim / num_heads, num_heads, attnRatio: 1, resolution: new Tuple<int, int>(window_size, window_size));
			this.mlp = new Mlp(inFeatures: dim, hiddenFeatures: (int)(dim * mlp_ratio), actLayer: activation);
			this.local_conv = new Conv2dBN(dim, dim, kernel_size: localConvSize, stride: 1, padding: localConvSize / 2, groups: dim);
			this.dropPathLayer = dropPath > 0 ? new DropPath(dropPath) : Identity();
			RegisterComponents();
		}

		public override Tensor forward(Tensor x)
		{
			using (NewDisposeScope())
			{
				var (H, W) = inputResolution;
				var shape = x.shape;
				var (B, L, C) = (shape[0], shape[1], shape[2]);
				Debug.Assert(L == H * W, "input feature has wrong size");
				Tensor resX = x;
				if (H == window_size && W == window_size)
				{
					x = attn.forward(x);
				}
				else
				{
					x = x.view(B, H, W, C);
					int padB = (window_size - H % window_size) % window_size;
					int padR = (window_size - W % window_size) % window_size;
					bool padding = padB > 0 || padR > 0;

					if (padding)
					{
						x = functional.pad(x, new long[] { 0, 0, 0, padR, 0, padB });
					}

					int pH = H + padB;
					int pW = W + padR;
					int nH = pH / window_size;
					int nW = pW / window_size;
					// window partition
					x = x.view(B, nH, window_size, nW, window_size, C).transpose(2, 3).reshape(B * nH * nW, window_size * window_size, C);
					x = attn.forward(x);
					// window reverse
					x = x.view(B, nH, nW, window_size, window_size, C).transpose(2, 3).reshape(B, pH, pW, C);

					if (padding)
					{
						x = x[TensorIndex.Slice(), TensorIndex.Slice(0, H), TensorIndex.Slice(0, W)].contiguous();
					}

					x = x.view(B, L, C);

				}
				x = resX + dropPathLayer.forward(x);

				x = x.transpose(1, 2).reshape(B, C, H, W);
				x = local_conv.forward(x);
				x = x.view(B, C, L).transpose(1, 2);

				x = x + dropPathLayer.forward(mlp.forward(x));
				return x.MoveToOuterDisposeScope();
			}
		}

		public string extra_repr()
		{
			return $"dim={dim}, input_resolution={inputResolution}, num_heads={num_heads}, " +
				   $"window_size={window_size}, mlp_ratio={mlp_ratio}";
		}
	}

	public class BasicLayer : Module<Tensor, Tensor>
	{
		private readonly ModuleList<TinyViTBlock> blocks;
		private readonly Module<Tensor, Tensor> downsample;
		private readonly bool useCheckpoint;
		private readonly long dim;
		private readonly Tuple<int, int> inputResolution;
		private readonly int depth;

		public BasicLayer(long dim, Tuple<int, int> inputResolution, int depth, int num_heads, int window_size, float mlp_ratio = 4, float drop = 0, List<float> dropPath = null, Func<Tuple<int, int>, long, long, Func<Module<Tensor, Tensor>>, Module<Tensor, Tensor>> downsample = null, bool useCheckpoint = false, int localConvSize = 3, Func<Module<Tensor, Tensor>> activation = null, long? out_dim = null, string name = "BasicLayer") : base(name)
		{
			this.useCheckpoint = useCheckpoint;
			this.dim = dim;
			this.inputResolution = inputResolution;
			this.depth = depth;
			if (dropPath is null)
			{
				dropPath = new List<float>() { 0 };
			}

			TinyViTBlock[] blockList = new TinyViTBlock[depth];

			for (int i = 0; i < depth; i++)
			{
				float dp = dropPath[0];
				if (dropPath.Count > 1)
				{
					dp = dropPath[i];
				}
				blockList[i] = new TinyViTBlock(dim, inputResolution, num_heads, window_size, mlp_ratio, drop, dp, localConvSize, activation);
			}

			blocks = new ModuleList<TinyViTBlock>(blockList);

			if (downsample is not null)
			{
				this.downsample = downsample(inputResolution, dim, out_dim.Value, activation);
			}
			RegisterComponents();
		}

		public override Tensor forward(Tensor x)
		{
			foreach (var blk in blocks)
			{
				x = blk.forward(x);
			}

			if (downsample is not null)
			{
				x = downsample.forward(x);
			}

			return x;
		}

		public string extra_repr()
		{
			return $"dim={dim}, input_resolution={inputResolution}, depth={depth}";
		}
	}

	//public class LayerNorm2d : Module<Tensor, Tensor>
	//{
	//	private readonly Tensor weight;
	//	private readonly Tensor bias;
	//	private readonly float eps;

	//	public LayerNorm2d(long numChannels, float eps = 1e-6, string name = "LayerNorm2d") : base(name)
	//	{
	//		this.eps = eps;
	//		weight = Parameter(ones(numChannels));
	//		bias = Parameter(zeros(numChannels));
	//		RegisterComponents();
	//	}

	//	public override Tensor forward(Tensor x)
	//	{
	//		using (NewDisposeScope())
	//		{
	//			Tensor u = x.mean(new long[] { 1 }, keepdim: true);
	//			Tensor s = (x - u).pow(2).mean(new long[] { 1 }, keepdim: true);
	//			x = (x - u) / torch.sqrt(s + eps);
	//			x = weight[TensorIndex.Slice(), TensorIndex.None, TensorIndex.None] * x + bias[TensorIndex.Slice(), TensorIndex.None, TensorIndex.None];
	//			return x.MoveToOuterDisposeScope();
	//		}
	//	}
	//}
	internal class TinyViT : Common.ImageEncoderViTBase
	{
		//public readonly int imgSize;
		private int _numClasses;
		private int[] _depths;
		private int _num_layers;
		private float _mlp_ratio;

		private PatchEmbed patch_embed;
		private ModuleList<Module<Tensor, Tensor>> layers;
		private LayerNorm norm_head;
		private Linear head;
		private Sequential neck;

		public TinyViT(int imgSize = 224, int inChans = 3, int numClasses = 1000,
					   int[] embed_dims = null, int[] depths = null, int[] num_heads = null,
					   int[] window_sizes = null, float mlp_ratio = 4f, float dropRate = 0f,
					   float dropPathRate = 0.1f, bool useCheckpoint = false,
					   float mbconvExpandRatio = 4.0f, int localConvSize = 3,
					   float layerLrDecay = 1.0f, string name = "TinyVit") : base(img_size: imgSize)
		{
			if (embed_dims == null)
			{
				embed_dims = new int[] { 96, 192, 384, 768 };
			}
			if (depths == null)
			{
				depths = new int[] { 2, 2, 6, 2 };
			}

			if (num_heads == null)
			{
				num_heads = new int[] { 3, 6, 12, 24 };
			}

			if (window_sizes == null)
			{
				window_sizes = new int[] { 7, 7, 14, 7 };
			}
			_numClasses = numClasses;
			_depths = depths;
			_num_layers = depths.Length;
			_mlp_ratio = mlp_ratio;

			patch_embed = new PatchEmbed(inChans, embed_dims[0], imgSize, GELU);

			long[] patchesResolution = patch_embed.PatchesResolution;

			float[] dpr = linspace(0, dropPathRate, depths.Sum()).data<float>().ToArray();


			layers = new ModuleList<Module<Tensor, Tensor>>();
			var downsample = (Tuple<int, int> a, long b, long c, Func<Module<Tensor, Tensor>> d) =>
			{
				return new PatchMerging(a, b, c, d);
			};
			for (int iLayer = 0; iLayer < _num_layers; iLayer++)
			{
				int dim = embed_dims[iLayer];
				Tuple<int, int> inputResolution = Tuple.Create(
					(int)(patchesResolution[0] / (int)Math.Pow(2, iLayer == 3 ? 2 : iLayer)),
					(int)(patchesResolution[1] / (int)Math.Pow(2, iLayer == 3 ? 2 : iLayer))
				);
				int depth = depths[iLayer];
				List<float> dropPath = (from d in dpr[depths[..iLayer].Sum()..depths[..(iLayer + 1)].Sum()] select (float)d).ToList();
				int out_dim = embed_dims[Math.Min(iLayer + 1, embed_dims.Length - 1)];

				if (iLayer == 0)
				{
					layers.Add(new ConvLayer(dim, inputResolution, depth, GELU, dropPath, downsample, out_dim: out_dim));
				}
				else
				{
					layers.Add(new BasicLayer(dim, inputResolution, depths[iLayer], num_heads[iLayer], window_sizes[iLayer], _mlp_ratio, dropRate, dropPath,
						iLayer < (_num_layers - 1) ? downsample : null, useCheckpoint, localConvSize, GELU, out_dim));
				}
			}

			// Classifier head
			norm_head = LayerNorm(embed_dims.Last());
			head = Linear(embed_dims.Last(), numClasses > 0 ? numClasses : 0);

			neck = Sequential(
				Conv2d(embed_dims.Last(), 256, kernel_size: 1, bias: false),
				new LayerNorm2d(256),
				Conv2d(256, 256, kernel_size: 3, padding: 1, bias: false),
				new LayerNorm2d(256)
			);
			RegisterComponents();
		}

		public Tensor ForwardFeatures(Tensor x)
		{
			x = patch_embed.forward(x);
			x = layers[0].forward(x);
			for (int i = 1; i < layers.Count; i++)
			{
				x = layers[i].forward(x);
			}
			var B = x.size(0);
			var C = x.size(2);
			x = x.view(B, 64, 64, C);
			x = x.permute(0, 3, 1, 2);
			x = neck.forward(x);
			return x;
		}

		public override Tensor forward(Tensor x)
		{
			using var _ = NewDisposeScope();
			x = ForwardFeatures(x);
			//x = norm_head.Forward(x);
			//x = head.Forward(x);
			return x.MoveToOuterDisposeScope();
		}
	}

	internal partial class Helper
	{
		public static TinyViT tiny_vit_5m_224(bool pretrained = false, int numClasses = 1000, float dropPathRate = 0.0f)
		{
			return new TinyViT(numClasses: numClasses, embed_dims: new int[] { 64, 128, 160, 320 }, depths: new int[] { 2, 2, 6, 2 },
				num_heads: new int[] { 2, 4, 5, 10 }, window_sizes: new int[] { 7, 7, 14, 7 }, dropPathRate: dropPathRate);
		}
		public static TinyViT tiny_vit_11m_224(bool pretrained = false, int numClasses = 1000, float dropPathRate = 0.0f)
		{
			return new TinyViT(numClasses: numClasses, embed_dims: new int[] { 64, 128, 256, 448 }, depths: new int[] { 2, 2, 6, 2 },
				num_heads: new int[] { 2, 4, 8, 14 }, window_sizes: new int[] { 7, 7, 14, 7 }, dropPathRate: dropPathRate);
		}
		public static TinyViT tiny_vit_21m_224(bool pretrained = false, int numClasses = 1000, float dropPathRate = 0.0f)
		{
			return new TinyViT(numClasses: numClasses, embed_dims: new int[] { 96, 192, 384, 576 }, depths: new int[] { 2, 2, 6, 2 },
				num_heads: new int[] { 3, 6, 12, 18 }, window_sizes: new int[] { 7, 7, 14, 7 }, dropPathRate: dropPathRate);
		}

		public static TinyViT tiny_vit_21m_384(bool pretrained = false, int numClasses = 1000, float dropPathRate = 0.0f)
		{
			return new TinyViT(imgSize: 384, numClasses: numClasses, embed_dims: new int[] { 96, 192, 384, 576 }, depths: new int[] { 2, 2, 6, 2 },
				num_heads: new int[] { 3, 6, 12, 18 }, window_sizes: new int[] { 12, 12, 24, 12 }, dropPathRate: dropPathRate);
		}

		public static TinyViT tiny_vit_21m_512(bool pretrained = false, int numClasses = 1000, float dropPathRate = 0.0f)
		{
			return new TinyViT(imgSize: 512, numClasses: numClasses, embed_dims: new int[] { 96, 192, 384, 576 }, depths: new int[] { 2, 2, 6, 2 },
				num_heads: new int[] { 3, 6, 12, 18 }, window_sizes: new int[] { 16, 16, 32, 16 }, dropPathRate: dropPathRate);
		}

	}
}
