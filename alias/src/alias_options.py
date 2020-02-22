class AliasOptions:

    def __init__(self, ow_coeff=False, ow_recon=False,
                 ow_pos=False, ow_intpos=False, ow_hist=False,
                 ow_dist=False):

        self.ow_coeff = ow_coeff
        self.ow_recon = ow_recon
        self.ow_pos = ow_pos
        self.ow_intpos = ow_intpos
        self.ow_hist = ow_hist

        if ow_hist:
            ow_dist = True

        self.ow_dist = ow_dist
