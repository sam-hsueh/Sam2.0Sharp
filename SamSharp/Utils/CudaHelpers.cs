using System;
using System.Reflection;

namespace Sam2Sharp.Utils
{
    public static class CudaHelpers
    {
        public static void TryEmptyCache()
        {
            try
            {
                var torchType = Type.GetType("TorchSharp.torch, TorchSharp");
                if (torchType != null)
                {
                    var cudaProp = torchType.GetProperty("cuda");
                    var cudaObj = cudaProp?.GetValue(null);
                    var emptyCacheMethod = cudaObj?.GetType().GetMethod("empty_cache");
                    emptyCacheMethod?.Invoke(cudaObj, null);
                }
            }
            catch { }
        }

        public static int? TryGetDeviceMajor(int deviceIndex)
        {
            try
            {
                var torchType = Type.GetType("TorchSharp.torch, TorchSharp");
                if (torchType != null)
                {
                    var cudaProp = torchType.GetProperty("cuda");
                    var cudaObj = cudaProp?.GetValue(null);
                    var getDevProps = cudaObj?.GetType().GetMethod("get_device_properties");
                    var props = getDevProps?.Invoke(cudaObj, new object[] { deviceIndex });
                    if (props != null)
                    {
                        var majorProp = props.GetType().GetProperty("major");
                        var majorVal = majorProp?.GetValue(props);
                        if (majorVal is int mi) return mi;
                    }
                }
            }
            catch { }
            return null;
        }
    }
}
