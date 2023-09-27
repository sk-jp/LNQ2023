from monai.transforms import MapTransform


class EnsureSameSized(MapTransform):
    def __init__(self, keys):
        assert(len(keys) == 2)
        super(EnsureSameSized, self).__init__(keys)

    def __call__(self, data):
        d = dict(data)
        vol0 = d[self.keys[0]]
        vol1 = d[self.keys[1]]
        
        # check the number of slices
        z0 = vol0.shape[-1]
        z1 = vol1.shape[-1]
        
        if z0 > z1:
            # crop the last slices in vol0
            vol0 = vol0[..., :z1]
            d[self.keys[0]] = vol0
        elif z0 < z1:
            # crop the last slices in vol1
            vol1 = vol1[..., :z0]
            d[self.keys[1]] = vol1
        
        return d
    