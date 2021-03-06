from gaussian_process import Space

space = Space({
    "dense":[{
        "dense":{
            "layers":[1,4],
            "units":(16,32,64,128,256),
            "activation":"relu"
        },
        "dropout":{
            "rate":(0.0, 0.1, 0.3, 0.5)
        }
    },
    {
        "dense":{
            "layers":[1,4],
            "units":(16,32,64,128,256),
            "activation":"relu"
        },
        "dropout":{
            "rate":(0.0, 0.1, 0.3, 0.5)
        }
    },
    {
        "dense":{
            "layers":[1,4],
            "units":(16,32,64,128,256),
            "activation":"relu"
        },
        "dropout":{
            "rate":(0.0, 0.1, 0.3, 0.5)
        }
    }]
})