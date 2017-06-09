from __future__ import print_function
import numpy as np
import deeplift
import deeplift.conversion.keras_conversion as kc

keras_model_weights="/srv/scratch/manyu/train/K562/ZBTB33_Seq_Meth_newModel/logdir_simple/model.weights.h5"
keras_model_json="/srv/scratch/manyu/train/K562/ZBTB33_Seq_Meth_newModel/logdir_simple/model.arch.json"
keras_model = kc.load_keras_model(weights=keras_model_weights,
                                  json=keras_model_json)


from deeplift.blobs import NonlinearMxtsMode
from collections import OrderedDict
method_to_model = OrderedDict()

for method_name, nonlinear_mxts_mode in [
    #The genomics default = rescale on conv layers, revealcance on fully-connected
    ('rescale_conv_revealcancel_fc', NonlinearMxtsMode.DeepLIFT_GenomicsDefault),
    ('rescale_all_layers', NonlinearMxtsMode.Rescale),
    ('revealcancel_all_layers', NonlinearMxtsMode.RevealCancel),
    ('grad_times_inp', NonlinearMxtsMode.Gradient),
    ('guided_backprop', NonlinearMxtsMode.GuidedBackprop)]:
    method_to_model[method_name] = kc.convert_functional_model(
        model=keras_model,
        nonlinear_mxts_mode=nonlinear_mxts_mode)