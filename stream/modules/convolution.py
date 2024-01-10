# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:32:08 2022
Modified on Tue Jan  9 17:47:18 2024

@author: Xiaohuai Le, Xiaobin Rong
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Union

"""
When export to ONNX format, ensure that the cache is saved as a tensor, not a list.
"""

class StreamConv1d(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                stride: int=1,
                padding: int=0,
                dilation: int=1,
                groups: int=1,
                bias: bool=True,
                *args, **kargs):
        super(StreamConv1d, self).__init__(*args, *kargs)
        
        assert padding == 0, "To meet the demands of causal streaming requirements"
        
        self.Conv1d = nn.Conv1d(in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                dilation = dilation,
                                groups = groups,
                                bias = bias)
    
    def forward(self, x, cache):
        """
        x:     [bs, C, T_size]
        cache: [bs, C, T_size-1]
        """
        inp = torch.cat([cache, x], dim=-1)
        oup = self.Conv1d(inp)
        out_cache = inp[..., 1:]
        return oup, out_cache


class StreamConv2d(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 *args, **kargs):
        super().__init__(*args, **kargs)
        """
        kernel_size = [T_size, F_size] by defalut
        """
        if type(padding) is int:
            self.T_pad = padding
            self.F_pad = padding
        elif type(padding) in [list, tuple]:
            self.T_pad, self.F_pad = padding
        else:
            raise ValueError('Invalid padding size.')
        
        assert self.T_pad == 0, "To meet the demands of causal streaming requirements"
        
        self.Conv2d = nn.Conv2d(in_channels = in_channels, 
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                dilation = dilation,
                                groups = groups,
                                bias = bias)
            
    def forward(self, x, cache):
        """
        x: [bs, C, 1, F]
        cache: [bs, C, T_size-1, F]
        """
        inp = torch.cat([cache, x], dim=2)
        outp = self.Conv2d(inp)
        out_cache = inp[:,:, 1:]
        return outp, out_cache


class StreamConvTranspose2d(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 *args, **kargs):
        super().__init__(*args, **kargs)
        """
        kernel_size = [T_size, F_size] by default
        stride = [T_stride, F_stride] and assert T_stride == 1
        """
        if type(kernel_size) is int:
            self.T_size = kernel_size
            self.F_size = kernel_size
        elif type(kernel_size) in [list, tuple]:
            self.T_size, self.F_size = kernel_size
        else:
            raise ValueError('Invalid kernel size.')
            
        if type(stride) is int:
            self.T_stride = stride
            self.F_stride = stride
        elif type(stride) in [list, tuple]:
            self.T_stride, self.F_stride = stride
        else:
            raise ValueError('Invalid stride size.')
        
        assert self.T_stride == 1

        if type(padding) is int:
            self.T_pad = padding
            self.F_pad = padding
        elif type(padding) in [list, tuple]:
            self.T_pad, self.F_pad = padding
        else:
            raise ValueError('Invalid padding size.')
        
        if type(dilation) is int:
            self.T_dilation = dilation
            self.F_dilation = dilation
        elif type(dilation) in [list, tuple]:
            self.T_dilation, self.F_dilation = dilation
        else:
            raise ValueError('Invalid dilation size.')
        
        assert self.T_pad == (self.T_size-1) * self.T_dilation, "To meet the demands of causal streaming requirements"

        self.ConvTranspose2d = nn.ConvTranspose2d(in_channels = in_channels, 
                                                out_channels = out_channels,
                                                kernel_size = kernel_size,
                                                stride = stride, 
                                                padding = padding,
                                                dilation = dilation,
                                                groups = groups,
                                                bias = bias)
        
    def forward(self, x, cache):
        """
        x: [bs, C, 1, F]
        cache: [bs, C, T_size-1, F]
        """
        inp = torch.cat([cache, x], dim=2)
        outp = self.ConvTranspose2d(inp)
        out_cache = inp[:,:, 1:]
        return outp, out_cache


if __name__ == '__main__':
    from convert import convert_to_stream

    ### test Conv1d Stream
    Sconv = StreamConv1d(1, 1, 3)
    Conv = nn.Conv1d(1, 1, 3)
    convert_to_stream(Sconv, Conv)

    test_input = torch.randn([1, 1, 10])
    with torch.no_grad():
        ## Non-Streaming
        test_out1 = Conv(torch.nn.functional.pad(test_input, [2,0]))
        
        ## Streaming
        cache = torch.zeros([1, 1, 2])
        test_out2 = []
        for i in range(10):
            out, cache = Sconv(test_input[..., i:i+1], cache)
            test_out2.append(out)
        test_out2 = torch.cat(test_out2, dim=-1)
        print(">>> Streaming Conv1d error:", (test_out1 - test_out2).abs().max())

    ### test Conv2d Stream
    Sconv = StreamConv2d(1, 1, [3,3])
    Conv = nn.Conv2d(1, 1, (3,3))
    convert_to_stream(Sconv, Conv)

    test_input = torch.randn([1,1,10,6])

    with torch.no_grad():
        ## Non-Streaming
        test_out1 = Conv(torch.nn.functional.pad(test_input,[0,0,2,0]))
        
        ## Streaming
        cache = torch.zeros([1,1,2,6])
        test_out2 = []
        for i in range(10):
            out, cache = Sconv(test_input[:,:, i:i+1], cache)
            test_out2.append(out)
        test_out2 = torch.cat(test_out2, dim=2)
        print(">>> Streaming Conv2d error:", (test_out1 - test_out2).abs().max())

        
    ### test ConvTranspose2d Stream
    kt = 3  # kernel size along T axis
    dt = 2  # dilation along T axis
    pt = (kt-1) * dt # padding along T axis
    DeConv = torch.nn.ConvTranspose2d(4, 8, (kt,3), stride=(1,2), padding=(pt,1), dilation=(dt,2), groups=2)
    SDeconv = StreamConvTranspose2d(4, 8, (kt,3), stride=(1,2), padding=(2*2,1), dilation=(dt,2), groups=2)
    convert_to_stream(SDeconv, DeConv)

    test_input = torch.randn([1, 4, 100, 6])
    with torch.no_grad():
        ## Non-Streaming
        test_out1 = DeConv(nn.functional.pad(test_input, [0,0,pt,0]))  # causal padding!
        test_out1 = test_out1
        ## Streaming
        test_out2 = []
        cache = torch.zeros([1, 4, pt, 6])
        for i in range(100):
            out, cache = SDeconv(test_input[:,:, i:i+1], cache)
            test_out2.append(out)
        test_out2 = torch.cat(test_out2, dim=2)

        print(">>> Streaming ConvTranspose2d error:", (test_out1 - test_out2).abs().max())
    

