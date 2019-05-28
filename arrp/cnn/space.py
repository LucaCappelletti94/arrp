from gaussian_process import Space

space = Space({
    "convolutionals":[{
        "Conv1D_kwargs":{
            "filters":[32, 512],
            "kernel_size":[5,20],
            "activation":"relu",
        },
        "MaxPooling1D_kwargs":{
            "pool_size":(2,)
        },
        "layers":[1,4]
    },
    {
        "Conv1D_kwargs":{
            "filters":[32, 512],
            "kernel_size":[5,20],
            "activation":"relu",
        },
        "MaxPooling1D_kwargs":{
            "pool_size":(2,)
        },
        "layers":[1,4]
    }],
    "dense":[{
        "layers":2,
        "units":32,
        "activation":"relu"
    },
    {
        "layers":2,
        "units":16,
        "activation":"relu"
    }]
})