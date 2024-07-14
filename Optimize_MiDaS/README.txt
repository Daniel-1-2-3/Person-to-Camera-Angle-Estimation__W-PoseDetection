Reason for quantization failure:

NotImplementedError: Could not run 'quantized::batch_norm2d' with arguments from the 'CPU' backend. 
This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). 
If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 
'quantized::batch_norm2d' is only available for these backends: 
[QuantizedCPU, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, 
Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, 
AutogradXLA, AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, AutogradMeta, Tracer, AutocastCPU, 
AutocastCUDA, FuncTorchBatched, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, 
PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].

qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
During quantization, configured the quantized model to run on backend qnnpack, but it was not supported by this CPU,
instead use backend fbgemm. 
IN ADDITION: quant stubs (input/output gates that convert floating point inputs to integers) were not added during traing

Pruning successful, 50% pruning used on all layers 