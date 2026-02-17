namespace Sam2Sharp_WinFormDemo
{
	partial class MainForm
	{
		/// <summary>
		///  Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		///  Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            groupBox1 = new GroupBox();
            PictureBox_Image = new PictureBox();
            groupBox3 = new GroupBox();
            PictureBox_Mask = new PictureBox();
            groupBox5 = new GroupBox();
            TextBox_ModelPath = new TextBox();
            ImageLoad = new Button();
            groupBox7 = new GroupBox();
            ComboBox_ScaleType = new ComboBox();
            Prov = new Button();
            Next = new Button();
            groupBox4 = new GroupBox();
            groupBox2 = new GroupBox();
            Models = new ComboBox();
            groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)PictureBox_Image).BeginInit();
            groupBox3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)PictureBox_Mask).BeginInit();
            groupBox5.SuspendLayout();
            groupBox7.SuspendLayout();
            groupBox4.SuspendLayout();
            groupBox2.SuspendLayout();
            SuspendLayout();
            // 
            // groupBox1
            // 
            groupBox1.Controls.Add(PictureBox_Image);
            groupBox1.Location = new Point(15, 14);
            groupBox1.Margin = new Padding(4);
            groupBox1.Name = "groupBox1";
            groupBox1.Padding = new Padding(4);
            groupBox1.Size = new Size(535, 500);
            groupBox1.TabIndex = 0;
            groupBox1.TabStop = false;
            groupBox1.Text = "Input Image";
            // 
            // PictureBox_Image
            // 
            PictureBox_Image.Dock = DockStyle.Fill;
            PictureBox_Image.Location = new Point(4, 24);
            PictureBox_Image.Margin = new Padding(4);
            PictureBox_Image.Name = "PictureBox_Image";
            PictureBox_Image.Size = new Size(527, 472);
            PictureBox_Image.SizeMode = PictureBoxSizeMode.Zoom;
            PictureBox_Image.TabIndex = 0;
            PictureBox_Image.TabStop = false;
            PictureBox_Image.MouseDown += PictureBox_Image_MouseDown;
            PictureBox_Image.MouseMove += PictureBox_Image_MouseMove;
            PictureBox_Image.MouseUp += PictureBox_Image_MouseUp;
            // 
            // groupBox3
            // 
            groupBox3.Controls.Add(PictureBox_Mask);
            groupBox3.Location = new Point(566, 14);
            groupBox3.Margin = new Padding(4);
            groupBox3.Name = "groupBox3";
            groupBox3.Padding = new Padding(4);
            groupBox3.Size = new Size(535, 500);
            groupBox3.TabIndex = 2;
            groupBox3.TabStop = false;
            groupBox3.Text = "Mask";
            // 
            // PictureBox_Mask
            // 
            PictureBox_Mask.Dock = DockStyle.Fill;
            PictureBox_Mask.Location = new Point(4, 24);
            PictureBox_Mask.Margin = new Padding(4);
            PictureBox_Mask.Name = "PictureBox_Mask";
            PictureBox_Mask.Size = new Size(527, 472);
            PictureBox_Mask.SizeMode = PictureBoxSizeMode.Zoom;
            PictureBox_Mask.TabIndex = 0;
            PictureBox_Mask.TabStop = false;
            // 
            // groupBox5
            // 
            groupBox5.Controls.Add(TextBox_ModelPath);
            groupBox5.Location = new Point(8, 24);
            groupBox5.Margin = new Padding(4);
            groupBox5.Name = "groupBox5";
            groupBox5.Padding = new Padding(4);
            groupBox5.Size = new Size(332, 67);
            groupBox5.TabIndex = 0;
            groupBox5.TabStop = false;
            groupBox5.Text = "Message";
            // 
            // TextBox_ModelPath
            // 
            TextBox_ModelPath.Location = new Point(8, 28);
            TextBox_ModelPath.Margin = new Padding(4);
            TextBox_ModelPath.Name = "TextBox_ModelPath";
            TextBox_ModelPath.ReadOnly = true;
            TextBox_ModelPath.Size = new Size(316, 27);
            TextBox_ModelPath.TabIndex = 0;
            // 
            // ImageLoad
            // 
            ImageLoad.Location = new Point(797, 47);
            ImageLoad.Margin = new Padding(4);
            ImageLoad.Name = "ImageLoad";
            ImageLoad.Size = new Size(131, 28);
            ImageLoad.TabIndex = 0;
            ImageLoad.Text = "Load Image";
            ImageLoad.UseVisualStyleBackColor = true;
            ImageLoad.Click += Button_ImageLoad_Click;
            // 
            // groupBox7
            // 
            groupBox7.Controls.Add(ComboBox_ScaleType);
            groupBox7.Location = new Point(620, 26);
            groupBox7.Margin = new Padding(4);
            groupBox7.Name = "groupBox7";
            groupBox7.Padding = new Padding(4);
            groupBox7.Size = new Size(157, 67);
            groupBox7.TabIndex = 5;
            groupBox7.TabStop = false;
            groupBox7.Text = "Scale Type";
            // 
            // ComboBox_ScaleType
            // 
            ComboBox_ScaleType.DropDownStyle = ComboBoxStyle.DropDownList;
            ComboBox_ScaleType.FormattingEnabled = true;
            ComboBox_ScaleType.Items.AddRange(new object[] { "Float32", "BFloat16" });
            ComboBox_ScaleType.Location = new Point(8, 22);
            ComboBox_ScaleType.Margin = new Padding(4);
            ComboBox_ScaleType.Name = "ComboBox_ScaleType";
            ComboBox_ScaleType.Size = new Size(135, 28);
            ComboBox_ScaleType.TabIndex = 1;
            ComboBox_ScaleType.SelectedIndexChanged += Models_SelectedIndexChanged;
            // 
            // Prov
            // 
            Prov.Location = new Point(946, 47);
            Prov.Margin = new Padding(4);
            Prov.Name = "Prov";
            Prov.Size = new Size(131, 28);
            Prov.TabIndex = 6;
            Prov.Text = "^";
            Prov.UseVisualStyleBackColor = true;
            Prov.Click += Prov_Click;
            // 
            // Next
            // 
            Next.Location = new Point(946, 85);
            Next.Margin = new Padding(4);
            Next.Name = "Next";
            Next.Size = new Size(131, 28);
            Next.TabIndex = 7;
            Next.Text = "v";
            Next.UseVisualStyleBackColor = true;
            Next.Click += Next_Click;
            // 
            // groupBox4
            // 
            groupBox4.Controls.Add(groupBox2);
            groupBox4.Controls.Add(Next);
            groupBox4.Controls.Add(Prov);
            groupBox4.Controls.Add(groupBox7);
            groupBox4.Controls.Add(ImageLoad);
            groupBox4.Controls.Add(groupBox5);
            groupBox4.Location = new Point(19, 533);
            groupBox4.Margin = new Padding(4);
            groupBox4.Name = "groupBox4";
            groupBox4.Padding = new Padding(4);
            groupBox4.Size = new Size(1085, 136);
            groupBox4.TabIndex = 3;
            groupBox4.TabStop = false;
            // 
            // groupBox2
            // 
            groupBox2.Controls.Add(Models);
            groupBox2.Location = new Point(348, 24);
            groupBox2.Margin = new Padding(4);
            groupBox2.Name = "groupBox2";
            groupBox2.Padding = new Padding(4);
            groupBox2.Size = new Size(254, 67);
            groupBox2.TabIndex = 8;
            groupBox2.TabStop = false;
            groupBox2.Text = "Model Type";
            // 
            // Models
            // 
            Models.DropDownStyle = ComboBoxStyle.DropDownList;
            Models.FormattingEnabled = true;
            Models.Location = new Point(8, 22);
            Models.Margin = new Padding(4);
            Models.Name = "Models";
            Models.Size = new Size(238, 28);
            Models.TabIndex = 1;
            Models.SelectedIndexChanged += Models_SelectedIndexChanged;
            // 
            // MainForm
            // 
            AutoScaleDimensions = new SizeF(9F, 20F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(1113, 709);
            Controls.Add(groupBox4);
            Controls.Add(groupBox3);
            Controls.Add(groupBox1);
            Margin = new Padding(4);
            Name = "MainForm";
            Text = "XJU Segment Anyting Model Sharp2.0";
            Load += MainForm_Load;
            groupBox1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)PictureBox_Image).EndInit();
            groupBox3.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)PictureBox_Mask).EndInit();
            groupBox5.ResumeLayout(false);
            groupBox5.PerformLayout();
            groupBox7.ResumeLayout(false);
            groupBox4.ResumeLayout(false);
            groupBox2.ResumeLayout(false);
            ResumeLayout(false);
        }

        #endregion

        private GroupBox groupBox1;
		private PictureBox PictureBox_Image;
		private GroupBox groupBox3;
		private PictureBox PictureBox_Mask;
        private GroupBox groupBox5;
        private TextBox TextBox_ModelPath;
        private Button ImageLoad;
        private GroupBox groupBox7;
        private ComboBox ComboBox_ScaleType;
        private Button Prov;
        private Button Next;
        private GroupBox groupBox4;
        private GroupBox groupBox2;
        private ComboBox Models;
    }
}
