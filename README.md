# onnx_MNN_C-
MNN  

https://www.yuque.com/mnn/cn/input

test two input mode 

1，
fill_data(img_data, pre_mask, change_ptr, index);
input_tensor_->copyFromHostTensor(nchwTensor);

2，
std::shared_ptr<CV::ImageProcess> pretreat(CV::ImageProcess::create(config));
pretreat->setMatrix(trans);
pretreat->convert((uint8_t*)inputImage, width, height, 0, input->writeMap<float>(), size_w, size_h, 4, 0, halide_type_of<float>());

