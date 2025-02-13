import torch
import torch.nn as nn
import pywt
from .wavelet_lowlevel import prep_filt_afb2d, mode_to_int, AFB2D

class DWT(nn.Module):
    """ Performs a Single-level 2d DWT Forward decomposition of an image

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
            
        mode (str): The padding scheme.
                    Support -> 'zero', 'symmetric', 'reflect' or 'periodization'. 
            
        supported wave:
            haar family: haar
            db family: db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12, db13, db14, db15, db16, db17, db18, db19, db20, db21, db22, db23, db24, db25, db26, db27, db28, db29, db30, db31, db32, db33, db34, db35, db36, db37, db38
            sym family: sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20
            coif family: coif1, coif2, coif3, coif4, coif5, coif6, coif7, coif8, coif9, coif10, coif11, coif12, coif13, coif14, coif15, coif16, coif17
            bior family: bior1.1, bior1.3, bior1.5, bior2.2, bior2.4, bior2.6, bior2.8, bior3.1, bior3.3, bior3.5, bior3.7, bior3.9, bior4.4, bior5.5, bior6.8
            rbio family: rbio1.1, rbio1.3, rbio1.5, rbio2.2, rbio2.4, rbio2.6, rbio2.8, rbio3.1, rbio3.3, rbio3.5, rbio3.7, rbio3.9, rbio4.4, rbio5.5, rbio6.8
            dmey family: dmey
            gaus family: gaus1, gaus2, gaus3, gaus4, gaus5, gaus6, gaus7, gaus8
            mexh family: mexh
            morl family: morl
            cgau family: cgau1, cgau2, cgau3, cgau4, cgau5, cgau6, cgau7, cgau8
            shan family: shan
            fbsp family: fbsp
            cmor family: cmor
    """
    def __init__(self, wave='haar', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.mode = mode

    def forward(self, x, only_low:bool = False, ignore_hh:bool=False):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        mode = mode_to_int(self.mode)
        w = x.size(-1)//2
        # Do 1 level of the transform
        # line 153 in lowlevel.py change to outsize = N//2
        x, high = AFB2D.apply(
            x, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
        
        if only_low:
            if x.size(-1) > w:
                no_pad = (x.size(-1)-w)//2
                x = x[:, :, no_pad:-no_pad, no_pad:-no_pad]
            return x
        elif ignore_hh:
            high = high[:,:,:2].contiguous()
            if x.size(-1) > w:
                no_pad = (x.size(-1)-w)//2
                x = x[:, :, no_pad:-no_pad, no_pad:-no_pad].contiguous()
                high = high[:, :, :, no_pad:-no_pad, no_pad:-no_pad].contiguous()
            return x, high
        else:
            return x, high