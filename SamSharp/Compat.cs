using System;
using System.Collections.Generic;

namespace SAM2Sharp
{
    // Lightweight compatibility shims to allow partial compilation.
    public class ndarray { public int[] Shape => new int[0]; }

    public class Union<T1, T2>
    {
        public object Value { get; }
        public Union(object v) { Value = v; }
        public bool Is<U>() => Value is U;
        public U As<U>() => (U)Value;
        public static implicit operator Union<T1, T2>(T1 v) => new Union<T1, T2>(v);
        public static implicit operator Union<T1, T2>(T2 v) => new Union<T1, T2>(v);
    }

    // Simple non-generic ModuleList placeholder used in code expecting ModuleList
    public class ModuleList : List<object>
    {
        public ModuleList() : base() { }
    }

    [AttributeUsage(AttributeTargets.Method)]
    public class TorchFunctionAttribute : Attribute
    {
        public TorchFunctionAttribute(string name) { }
    }

    // PIL image placeholder
    public class PILImage
    {
        public int Width { get; set; }
        public int Height { get; set; }
    }
}
