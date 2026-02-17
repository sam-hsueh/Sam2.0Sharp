using OpenCvSharp;
using Sam2Sharp.Utils;
using SkiaSharp;
using System;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using TorchSharp;
using SAM2Sharp;
using static Sam2Sharp.Utils.Classes;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;
using static TorchSharp.torch;

namespace Sam2Sharp_WinFormDemo
{
    public partial class MainForm : Form
    {
        private SKBitmap image;
        private Image pictureBoxImage;
        private float scale = 1.0f;
        private int padX = 0;
        private int padY = 0;
        private SAM2Sharp.SAM2ImagePredictor predictor;
        private SAM2AutomaticMaskGenerator sam;
        //private string modelPath;
        string BaseModelPath = "D:\\Models\\sam";

        //private List<SamPoint> UsamPoints = new List<SamPoint>();
        private List<SamPoint> DsamPoints
        {
            get => new List<SamPoint>(){ DLP, DRP};
        }

        public MainForm()
        {
            InitializeComponent();
            string[] ModelNames = new string[]
            {
                //"ObjectAwareModel.pt",
                "sam2.1_hiera_tiny.pt",
                "sam2_hiera_tiny.pt",
                "sam2.1_hiera_small.pt",
                "sam2_hiera_small.pt",
                "sam2.1_hiera_base_plus.pt"
            };
            Models.DataSource = ModelNames;
            //Models.SelectedIndex = 0;
            //            files = Directory.GetFiles(@"D:\WeldImages\160X120X3", "*.png");
             files = Directory.GetFiles(@"D:\WeldImages\320X240X3", "*.png");
          //  files = Directory.GetFiles(@"D:\WeldImages\520X390X3", "*.png");
            //files = Directory.GetFiles(@"D:\ImageFeatureAnnotation\MecaVision", "*.jpg");
            // files = Directory.GetFiles(@"D:\Tracking\CI\SAW\NGTracking\bin\Debug\net9.0-windows10.0.22000.0\2025-11-03", "*.png");
            index = 0;
            // 加载权重
         //   string weights_path = "D:\\models\\sam\\Prompt_guided_Mask_Decoder.pt";
            var image = Cv2.ImRead(files[index]);
            DLP = new SamPoint(40, image.Height - 40, true);
            DRP = new SamPoint(image.Width - 40, image.Height - 40, true);

        }
        string[] files;
        int index = -1;
        bool isMovedFp = false;
        private void Button_ImageLoad_Click(object sender, EventArgs e)
        {
            if (predictor is null)
            {
                MessageBox.Show("Model not loaded.");
                return;
            }

            //if (sam is null)
            //{
            //	MessageBox.Show("Model not loaded.");
            //	return;
            //}
            if (index == -1 || files == null)
                return;
            try
            {
                string filePath = files[index];
                //filePath=Path.Combine(@"D:\WeldImages\320X240X3", "12-20-20.png");
                PictureBox_Image.Image = Image.FromFile(filePath);
                pictureBoxImage = Image.FromFile(filePath);
                image = SKBitmap.Decode(filePath);
                int w = image.Width;
                int h = image.Height;
                int pictureBoxWidth = PictureBox_Image.Width;
                int pictureBoxHeight = PictureBox_Image.Height;


                scale = Math.Min((float)pictureBoxWidth / w, (float)pictureBoxHeight / h);

                padX = (pictureBoxWidth - (int)(w * scale)) / 2;
                padY = (pictureBoxHeight - (int)(h * scale)) / 2;

                //var imageMat = Cv2.ImRead(filePath);
                long start = DateTime.Now.Ticks;
                //List<string> list = new List<string>();
                //list.Add(filePath);
                //list.Add(files[index + 1 == files.Length ? 0 : index + 1]);
                if (predictor != null&&isMovedFp==false)
                {
                    var mat = Cv2.ImRead(filePath, ImreadModes.Color);
                    predictor.set_image(mat);

                    //predictor.set_image(filePath);
                    //                    predictor.set_image_batch(list);
                }
                long end = DateTime.Now.Ticks;
                long GIelapsedMs = (end - start) / TimeSpan.TicksPerMillisecond;

                //start = DateTime.Now.Ticks;
                if (image is null)
                {
                    MessageBox.Show("Please load an image first.");
                    return;
                }
                //DsamPoints = new List<SamPoint>();
                //DLP = new SamPoint(40, image.Height - 40, true);
                //DRP = new SamPoint(image.Width - 40, image.Height - 40, true);
                //ULP = new SamPoint(40, 40, false);
                //URP = new SamPoint(image.Width - 40, 40, false);


                //DsamPoints.Add(ULP);
                //DsamPoints.Add(URP);
                //DsamPoints.Add(DLP);
                //DsamPoints.Add(DRP);
                DrawPointsAndBoxes();

                //var all_DsamPoints = new List<List<SamPoint>>();
                //for(int p=0;p<list.Count;p++)
                //{
                //    all_DsamPoints.Add(DsamPoints);
                //}
                //List<List<PredictOutput>> all_Doutputs = null;
                //if (predictor != null)
                //    all_Doutputs = predictor.Predict_Batch(all_DsamPoints);
                //var Doutputs = all_Doutputs[0];

                List<PredictOutput> Doutputs = null;
                if (predictor != null)
 //                   Doutputs = predictor.Predict(DsamPoints);
                Doutputs = predictor.Predict(point_labels);
                //if (sam != null)
                //    Doutputs = sam.generate(image);
                long end1 = DateTime.Now.Ticks;
                long GIelapsedMs1 = (end1 - end) / TimeSpan.TicksPerMillisecond;
                //TextBox_ModelPath.Text = GIelapsedMs.ToString();


                SKBitmap resultBmp = image.Copy();
                using (SKCanvas canvas = new SKCanvas(resultBmp))
                {
                    Random random = new Random();
                    SKColor color = new SKColor((byte)random.Next(256), (byte)random.Next(256), (byte)random.Next(256), 90);
                    SKPaint skp = new SKPaint();
                    skp.Color = new SKColor(0, 255, 0);
                    skp.StrokeWidth = 2;
                    for (int i = 0; i < Doutputs.Count; i++)
                    {
                        SKImageInfo info = new SKImageInfo(w, h);
                        using (SKBitmap maskBitmap = new SKBitmap(info))
                        using (SKCanvas c = new SKCanvas(maskBitmap))
                        {
                            c.Clear(SKColors.Transparent);
                            PredictOutput output = Doutputs[i];

                            // Precision of the masks
                            Console.WriteLine($"Mask {i}: Precision: {output.Precision * 100:F2}%");
                            bool[,] mask = output.Mask;

                            var src = new Mat(w, h, MatType.CV_8UC1);
                            int length = w * h; // or src.Height * src.Step;

                            byte[] mask1 = new byte[length];
                            Buffer.BlockCopy(mask, 0, mask1, 0, length);
                            Marshal.Copy(mask1, 0, src.Data, length);
                            Mat labelMat = new Mat();
                            Mat stats = new Mat();
                            Mat centroids = new Mat();
                            num = Cv2.ConnectedComponentsWithStats(src, labelMat, stats, centroids, PixelConnectivity.Connectivity8);
                            var sizes = new List<int>();
                            for (int j = 1; j < num; j++)
                            {
                                sizes.Add(stats.At<int>(j, 4));
                            }
                            List<int> numbers = Enumerable.Range(1, num - 1).Select(k => k).ToList<int>();  // 可在此处扩展逻辑，比如 i * 2、(float)i 等
                            // 6. 筛选出面积小于阈值的小区域标签
                            var area_thresh = 800;
                            //var small_regions = new List<int>();
                            for (int j = 0; j < sizes.Count; j++)
                            {
                                if (sizes[j] < area_thresh)
                                {
                                    //small_regions.Add(j + 1); // 对应连通域标签（从1开始）
                                    numbers.Remove(j + 1);
                                }
                            }
                            int[] array;
                            labelMat.GetArray(out array);
                            //var tlist = new List<(int, int)>();
                            //for (int j = 0; j < num - 1; j++)
                            //{
                            //    int cc = array.Where(t => t == j + 1).Count();
                            //    tlist.Add((j + 1, cc));
                            //}
                            //var b = tlist.OrderByDescending(t => t.Item2).ToList();
                            //List<int> tmp = new List<int>();
                            //for (int j = 0; j < Math.Min(b.Count, 2); j++)
                            //{
                            //    if (j == 0 || (j > 0 && b[j].Item2 > 800))
                            //    { tmp.Add(b[j].Item1); }
                            //}
                            for (int y = 0; y < mask.GetLength(1); y++)
                            {
                                for (int x = 0; x < mask.GetLength(0); x++)
                                {
                                    int v = array[x * mask.GetLength(1) + y];
                                    if (v > 0 && numbers.Contains(array[x * mask.GetLength(1) + y]))
                                    {
                                        c.DrawPoint(x, y, color);
                                    }
                                }
                            }

                            for (int j = 0; j < numbers.Count; j++)
                            {
                                byte[] emask = new byte[length];
                                for (int l = 0; l < length; l++)
                                {
                                    if (array[l] == numbers[j])
                                        emask[l] = 1;
                                }
                                var src2 = new Mat(w, h, MatType.CV_8UC1);
                                Marshal.Copy(emask, 0, src2.Data, length);
                                OpenCvSharp.Point[][] contours;
                                Cv2.FindContours(
                                    image: src2,
                                    contours: out contours,
                                    hierarchy: out HierarchyIndex[] outputArray,
                                    mode: (RetrievalModes)RetrievalModes.External,
                                    method: (ContourApproximationModes)ContourApproximationModes.ApproxSimple
                                    );
                                List<OpenCvSharp.Point[]> query = contours.ToList<OpenCvSharp.Point[]>().OrderByDescending(t => Cv2.ContourArea(t)).Select(t => t).Take(1).ToList();
                                if (query.Count == 0)
                                    return;
                                var epsilon = Cv2.ArcLength(query[0], true) * 0.0015;
                                var approxContour = Cv2.ApproxPolyDP(query[0], epsilon, true);
                                var curPS = new List<SKPoint>();
                                for (int k = 0; k < approxContour.Length; k++)
                                {
                                    curPS.Add(new SKPoint((float)(approxContour[k].Y), (float)(approxContour[k].X)));
                                    c.DrawRect((float)(approxContour[k].Y - 3), (float)(approxContour[k].X - 3), 6, 6, skp);
                                }
                                curPS.Add(curPS[0]);
                                c.DrawPoints(SKPointMode.Polygon, curPS.ToArray(), skp);
                            }
                            canvas.DrawBitmap(maskBitmap, new SKPoint(0, 0));
                        }
                    }
                }
                PictureBox_Mask.Image = SKBitmapToBitmap(resultBmp);
                GC.Collect();
                TextBox_ModelPath.Text = GIelapsedMs.ToString() + "，"+ GIelapsedMs1.ToString() + "，" + num + "," + (new FileInfo(files[index])).Name;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading image: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void DrawPointsAndBoxes()
        {
            if (pictureBoxImage is null)
            {
                MessageBox.Show("Please load an image first.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            Image img = pictureBoxImage.Clone() as Image;
            Graphics graphics = Graphics.FromImage(img);
            //UsamPoints.ForEach(point =>
            //{
            //	graphics.FillEllipse(Brushes.Green, point.X - 10, point.Y - 10, 20, 20);
            //});

            DsamPoints.ForEach(point =>
            {
                graphics.FillEllipse(Brushes.Red, point.X - 5, point.Y - 5, 10, 10);
            });
            graphics.Dispose();
            PictureBox_Image.Image = img;
        }
        int num = 0;

        private void MainForm_Load(object sender, EventArgs e)
        {
            Models.SelectedIndex = 0;
            ComboBox_ScaleType.SelectedIndex = 1;
        }

        private Bitmap SKBitmapToBitmap(SKBitmap skBitmap)
        {
            using (var stream = new MemoryStream())
            {
                skBitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
                stream.Seek(0, SeekOrigin.Begin);
                return new Bitmap(stream);
            }
        }

        //private void Button_ModelLoad_Click(object sender, EventArgs e)
        //{
        //    if (string.IsNullOrEmpty(modelPath))
        //    {
        //        MessageBox.Show("Please Load a model first");
        //    }
        //    //   Sam2Device device = (Sam2Device)Enum.Parse(typeof(Sam2Device), ComboBox_Device.Text);  // SamDevice.Cuda or SamDevice.
        //    torch.ScalarType dtype = (torch.ScalarType)Enum.Parse(typeof(torch.ScalarType), ComboBox_ScaleType.Text); // Float, Half or BF16.
        //    try
        //    {
        //        predictor = new SAM2Sharp.SAM2ImagePredictor(modelPath, torch.CUDA, dtype);
        //        //sam = new SamAutomaticMaskGenerator(modelPath);
        //        MessageBox.Show("Model Loaded Done.");
        //    }
        //    catch (Exception ex)
        //    {
        //        MessageBox.Show(ex.Message);
        //    }
        //}
        SamPoint DLP
        {
            set; get;
        } = new SamPoint(40, 200, true);
        SamPoint DRP
        {
            set; get;
        } = new SamPoint(280, 200, true);
        SamPoint ULP
        {
            set; get;
        } = new SamPoint(40, 40, false);
        SamPoint URP
        {
            set; get;
        } = new SamPoint(280, 40, false);


        private void PictureBox_Image_MouseDown(object sender, MouseEventArgs e)
        {
            if (!mdown && hover)
            {
                mdown = true;
                var x = (e.X - padX) / scale;
                var y = (e.Y - padY) / scale;
                var dlp = (DLP.X - x) * (DLP.X - x) + (DLP.Y - y) * (DLP.Y - y);
                var drp = (DRP.X - x) * (DRP.X - x) + (DRP.Y - y) * (DRP.Y - y);
                var minv = Math.Min(dlp, drp);
                if (minv == dlp)
                    flag = 0;
                else if (minv == drp)
                    flag = 1;
            }
        }
        public static bool hover = false;
        bool mdown = false;
        private void PictureBox_Image_MouseUp(object sender, MouseEventArgs e)
        {
            flag = -1;
            mdown = false;
            if(isMovedFp)
            {
                point_labels = _prep_prompts(DsamPoints);
            }
        }
        int flag = -1;
        private void PictureBox_Image_MouseMove(object sender, MouseEventArgs e)
        {
            var dlp = DLP;
            var drp = DRP;
            if (!mdown)
            {
                var p = new OpenCvSharp.Point((e.X - padX) / scale, (e.Y - padY) / scale);
                if ((p.X < 10.0 + dlp.X && p.X > -10.0 + dlp.X) && (p.Y < 10.0 + dlp.Y && p.Y > -10.0 + dlp.Y) || (p.X < 10.0 + drp.X && p.X > -10.0 + drp.X) && (p.Y < 10.0 + drp.Y && p.Y > -10.0 + drp.Y))
                {
                    PictureBox_Image.Cursor = Cursors.SizeAll;
                    hover = true;
                }
                else
                {
                    PictureBox_Image.Cursor = Cursors.Hand;
                    hover = false;
                }
                flag = -1;
            }
            else
            {
                if (e.Button == MouseButtons.Left)
                {
                    if (flag == 0)
                    {
                        int dx = (int)((e.X - padX) / scale - dlp.X);
                        int dy = (int)((e.Y - padY) / scale - dlp.Y);
                        DLP.X += dx;
                        DLP.Y += dy;
                    }
                    else if (flag == 1)
                    {
                        int dx = (int)((e.X - padX) / scale - drp.X);
                        int dy = (int)((e.Y - padY) / scale - drp.Y);
                        DRP.X += dx;
                        DRP.Y += dy;
                    }
                    isMovedFp = true;
                    DrawPointsAndBoxes();
                }
                else
                {
                    mdown = false;
                }
            }
        }

        private void Prov_Click(object sender, EventArgs e)
        {
            if (files != null && files.Length > 0)
            {
                index--;
                if (index <= 0)
                    index = files.Length - 1;
                isMovedFp = false;
                ImageLoad.PerformClick();
            }
        }

        private void Next_Click(object sender, EventArgs e)
        {
            if (files != null && files.Length > 0)
            {
                index++;
                if (index >= files.Length)
                    index = 0;
                isMovedFp = false;
                ImageLoad.PerformClick();
            }
        }

        private void Models_SelectedIndexChanged(object sender, EventArgs e)
        {
            string modelPath = BaseModelPath +"\\"+ Models.SelectedItem.ToString();
            if (string.IsNullOrEmpty(modelPath))
            {
                MessageBox.Show("Please Load a model first");
            }
            if (ComboBox_ScaleType.Text == "")
                return;
            //   Sam2Device device = (Sam2Device)Enum.Parse(typeof(Sam2Device), ComboBox_Device.Text);  // SamDevice.Cuda or SamDevice.
            string dtypeStr = ComboBox_ScaleType.Text==""? "BFloat16": ComboBox_ScaleType.Text;
            dtype = (torch.ScalarType)Enum.Parse(typeof(torch.ScalarType), dtypeStr); // Float, Half or BF16.
            device = torch.CUDA;
            try
            {
                predictor = new SAM2Sharp.SAM2ImagePredictor(modelPath, device, dtype);
                //sam = new SamAutomaticMaskGenerator(modelPath);
                MessageBox.Show("Model Loaded Done.");
                isMovedFp = false;
                point_labels = _prep_prompts(DsamPoints);
                ImageLoad.PerformClick();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }

        }

        (Tensor points,Tensor labels) point_labels;
        ScalarType dtype;
        torch.Device device;
        private (Tensor, Tensor) _prep_prompts(List<SamPoint> points)
        {
            var image = Cv2.ImRead(files[index]);
            Tensor pointcoords = null, labels = null;
            if (points is not null)
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
             //   var coords = pointcoords;
                //if (normalizeCoords)
                {
                    //var xSlice = unnorm_coords.index_select(-1, tensor(0));  // 取出所有X坐标
                    //unnorm_coords.index_put_(new TensorIndex[] { TensorIndex.Ellipsis, tensor(0) }, xSlice / orig_hw[0].Item2);

                    //// 步骤3：对最后一维的第1个元素（Y坐标）除以h
                    //var ySlice = unnorm_coords.index_select(-1, tensor(1));  // 取出所有Y坐标
                    //unnorm_coords.index_put_(new TensorIndex[] { TensorIndex.Ellipsis, tensor(1) },  ySlice / orig_hw[0].Item2);
                    pointcoords[.., 0] = pointcoords[.., 0] / image.Width;
                    pointcoords[.., 1] = pointcoords[.., 1] / image.Height;
                  //  pointpointcoords = pointcoords;
                }
                pointcoords = pointcoords * predictor.model.image_size;
                pointcoords = pointcoords[TensorIndex.None, ..];
                labels = labels[TensorIndex.None, ..];
                //unnorm_coords.unsqueeze(0);
                //labels.unsqueeze(0);
            }
            return (pointcoords, labels);
        }
    }
}
