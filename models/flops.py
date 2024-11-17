from ptflops import get_model_complexity_info


def get_operations(model):
    macs, params = get_model_complexity_info(model, (3, 800, 800), 
                                             as_strings=True, backend='pytorch',
                                             print_per_layer_stat=False, verbose=False)
    
    flops = 2 * float(macs.split()[0])
    units = {
        "GMac": "GFlops",
        "MMac": "MFlops",
    }
    flops = f'{flops:.4f} {units[macs.split()[1].strip()]}'
    return {
        "params": params,
        "MACS": macs,  # Multiply-Add Operations per Second
        "FLOPS": flops,  # Floating-Point Operations per Second
    }