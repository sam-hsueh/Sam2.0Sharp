using TorchSharp;
using TorchSharp.Modules;
using static Sam2Sharp.Modeling.Sam2.Transformer;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
namespace Sam2Sharp.Modeling
{
    public class MemoryAttentionLayer : Module<Tensor, Tensor, Tensor, Tensor, int,Tensor>
	{
        public int d_model;
        public int dim_feedforward;
        public float dropout_value;
        public RoPEAttention self_attn;
        public RoPEAttention cross_attn_image;
        public Linear linear1;
        public Dropout dropout;
        public Linear linear2;
        public LayerNorm norm1;
        public LayerNorm norm2;
        public LayerNorm norm3;
        public Dropout dropout1;
        public Dropout dropout2;
        public Dropout dropout3;
        public string activation_str;
        public Func<Tensor,bool, Tensor> activation;
        public bool pos_enc_at_attn;
        public bool pos_enc_at_cross_attn_queries;
        public bool pos_enc_at_cross_attn_keys;
        public readonly float drop_pathf;


        public MemoryAttentionLayer(
            Func<Tensor, bool, Tensor> activation,
            RoPEAttention cross_attention,
            int d_model,
            int dim_feedforward,
            float dropout,
            bool pos_enc_at_attn,
            bool pos_enc_at_cross_attn_keys,
            bool pos_enc_at_cross_attn_queries,
            RoPEAttention self_attention) : base("MemoryAttentionLayer")
        {
            this.d_model = d_model;
            this.dim_feedforward = dim_feedforward;
            this.dropout_value = dropout;
            this.self_attn = self_attention;
            this.cross_attn_image = cross_attention;
            this.drop_pathf = dropout;
            // 前馈网络实现
            this.linear1 = Linear(d_model, dim_feedforward);
            this.dropout = Dropout(dropout);
            this.linear2 = Linear(dim_feedforward, d_model);

            this.norm1 = LayerNorm(d_model);
            this.norm2 = LayerNorm(d_model);
            this.norm3 = LayerNorm(d_model);
            this.dropout1 = Dropout(dropout);
            this.dropout2 = Dropout(dropout);
            this.dropout3 = Dropout(dropout);

            //this.activation_str = activation.ToString();
            this.activation = /*GetActivationFunction*/(activation);

            // 位置编码添加位置
            this.pos_enc_at_attn = pos_enc_at_attn;
            this.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries;
            this.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys;

            RegisterComponents();
        }

        private Tensor _forward_sa(Tensor tgt, Tensor query_pos)
        {
            // 自注意力
            var tgt2 = norm1.forward(tgt);
            var q = pos_enc_at_attn ? tgt2 + query_pos : tgt2;
            var k = pos_enc_at_attn ? tgt2 + query_pos : tgt2;
            tgt2 = self_attn.forward(q, k, tgt2,0);
            tgt = tgt + dropout1.forward(tgt2);
            return tgt;
        }

        private Tensor _forward_ca(Tensor tgt, Tensor memory, Tensor query_pos, Tensor pos, int num_k_exclude_rope = 0)
        {
            var kwds = new Dictionary<string, object>();
            if (num_k_exclude_rope > 0)
            {
                if (!(cross_attn_image is RoPEAttention))
                    throw new InvalidOperationException("cross_attn must be RoPEAttention when num_k_exclude_rope > 0");
                kwds["num_k_exclude_rope"] = num_k_exclude_rope;
            }

            // 交叉注意力
            var tgt2 = norm2.forward(tgt);
            var q = pos_enc_at_cross_attn_queries ? tgt2 + query_pos : tgt2;
            var k = pos_enc_at_cross_attn_keys ? memory + pos : memory;

            tgt2 = cross_attn_image.forward(q, k, memory, num_k_exclude_rope);
            tgt = tgt + dropout2.forward(tgt2);
            return tgt;
        }

        public override Tensor forward(Tensor tgt, Tensor memory, Tensor pos = null, Tensor query_pos = null, int num_k_exclude_rope = 0)
        {
            // 自注意力、交叉注意力
            tgt = _forward_sa(tgt, query_pos);
            tgt = _forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope);

            // MLP
            var tgt2 = norm3.forward(tgt);
            tgt2 = linear2.forward(dropout.forward(activation(linear1.forward(tgt2),false)));
            tgt = tgt + dropout3.forward(tgt2);

            return tgt;
        }

        private Func<Tensor,bool, Tensor> GetActivationFunction(string activation)
        {
            return activation.ToLower() switch
            {
                "relu" => functional.relu,
                "gelu" => functional.gelu,
                "silu" => functional.silu,
                "tanh" => functional.tanh,
                _ => throw new ArgumentException($"不支持的激活函数: {activation}")
            };
        }
    }

    public class MemoryAttention : Module<Tensor, Tensor, Tensor, Tensor, int, Tensor>
    {
        public int d_model;
        public ModuleList<MemoryAttentionLayer> layers;
        public int num_layers;
        public LayerNorm norm;
        public bool pos_enc_at_input;
        public bool batch_first;

        public MemoryAttention(
            int d_model,
            bool pos_enc_at_input,
            MemoryAttentionLayer layer,
            int num_layers,
            bool batch_first = true) : base("MemoryAttention")
        {
            this.d_model = d_model;
            layers = new ModuleList<MemoryAttentionLayer>();
            layers.Add(layer);
            for (int i = 0; i < num_layers-1; i++)
            {
            //    var newLayer = new MemoryAttentionLayer(
            //    activation: functional.relu,
            //    cross_attention: new RoPEAttention(
            //embedding_dim: layer.cross_attn_image.embedding_dim,
            //num_heads: layer.cross_attn_image.num_heads,
            //downsample_rate: layer.cross_attn_image.downsample_rate,
            //dropout: layer.cross_attn_image.dropout,
            // kv_in_dim: layer.cross_attn_image.kv_in_dim,
            // rope_theta: layer.cross_attn_image.rope_theta,
            // rope_k_repeat: layer.cross_attn_image.rope_k_repeat,
            // feat_sizes: layer.cross_attn_image.feat_sizes),
            //   d_model: layer.d_model,
            //   dim_feedforward: layer.dim_feedforward,
            //  dropout: layer.drop_pathf,
            //  pos_enc_at_attn: layer.pos_enc_at_attn,
            //  pos_enc_at_cross_attn_keys: layer.pos_enc_at_cross_attn_keys,
            //  pos_enc_at_cross_attn_queries: layer.pos_enc_at_cross_attn_queries,
            //  self_attention: new RoPEAttention(
            //   rope_theta: layer.self_attn.rope_theta,
            //   feat_sizes: layer.self_attn.feat_sizes,
            //   embedding_dim: layer.self_attn.embedding_dim,
            //   num_heads: layer.self_attn.num_heads,
            //   downsample_rate: layer.self_attn.downsample_rate,
            //   dropout: layer.self_attn.dropout)
            //    );
                layers.Add(layer);
            }
            this.num_layers = num_layers;
            this.norm = LayerNorm(d_model);
            this.pos_enc_at_input = pos_enc_at_input;
            this.batch_first = batch_first;
            RegisterComponents();
        }

        public override Tensor forward(Tensor curr, Tensor memory, Tensor curr_pos = null, Tensor memory_pos = null, int num_obj_ptr_tokens = 0)
        {
            //// 处理可能的列表输入（如果需要）
            //var currList = curr as List<Tensor>;
            //if (currList != null)
            //{
            //    var posList = curr_pos as List<Tensor>;
            //    if (posList == null || currList.Count != posList.Count || currList.Count != 1)
            //        throw new ArgumentException("curr和curr_pos必须是长度为1的列表");

            //    curr = currList[0];
            //    curr_pos = posList[0];
            //}

            //if (curr.size(1) != memory.size(1))
            //    throw new ArgumentException("curr和memory的批次大小必须相同");

            var output = curr;
            if (pos_enc_at_input && curr_pos is not null)
                output = output + 0.1f * curr_pos;

            Tensor originalCurrPos = curr_pos;
            Tensor originalMemory = memory;
            Tensor originalMemoryPos = memory_pos;

            if (batch_first)
            {
                // 转换为batch first格式
                output = output.transpose(0, 1);
                curr_pos = curr_pos?.transpose(0, 1);
                memory = memory.transpose(0, 1);
                memory_pos = memory_pos?.transpose(0, 1);
            }

            foreach (var layer in layers.Cast<MemoryAttentionLayer>())
            {
                var kwds = new Dictionary<string, object>();
                if (layer.cross_attn_image is RoPEAttention)
                    kwds["num_k_exclude_rope"] = num_obj_ptr_tokens;

                output = layer.forward(output, memory, memory_pos, curr_pos,
                    kwds.ContainsKey("num_k_exclude_rope") ? (int)kwds["num_k_exclude_rope"] : 0);
            }

            var normed_output = norm.forward(output);

            if (batch_first)
            {
                // 转换回seq first格式
                normed_output = normed_output.transpose(0, 1);
                curr_pos = originalCurrPos;
            }

            return normed_output;
        }
    }
}