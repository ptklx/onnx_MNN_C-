#include <vector>
#include <iostream>
#include <string>
#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>

#pragma comment(lib, "MNN_debug.lib")

class onnxSegmentation {
public:


	onnxSegmentation();
	~onnxSegmentation();
	int Init(const char* root_path);
	int Detect(cv::Mat img_data, cv::Mat pre_mask, cv::Mat& out_mask);

private:
	bool initialized_;
	//const cv::Size inputSize_ = {640, 384 };
	std::vector<int> dims_ = { 1, 4, 384, 640 };  //
	//const float meanVals_[4] = { 0.5f, 0.5f, 0.5f,0 };
	//const float normVals_[4] = { 0.007843f, 0.007843f, 0.007843f,1 };
	//float meanVals_[4] = { 127.5f, 127.5f, 127.5f,125 };
	//float normVals_[4] = { 0.00785f, 0.00785f, 0.00785f,1 };

	std::shared_ptr<MNN::Interpreter> onnx_interpreter_;
	MNN::Session* onnxSegmentation_sess_ = nullptr;

	MNN::Tensor* input_tensor_ = nullptr;
	std::shared_ptr<MNN::CV::ImageProcess> pretreat_data_ = nullptr;
	MNN::Tensor* nchwTensor = nullptr;
};



onnxSegmentation::onnxSegmentation() {
	initialized_ = false;
}

onnxSegmentation::~onnxSegmentation() {
	onnx_interpreter_->releaseModel();
	onnx_interpreter_->releaseSession(onnxSegmentation_sess_);
	delete nchwTensor;
}

int onnxSegmentation::Init(const char* root_path) {
	std::cout << "start Init." << std::endl;
	std::string model_file = std::string(root_path); //std::string(root_path) + "/onnxSegmentation.mnn";

	onnx_interpreter_ = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
	if (nullptr == onnx_interpreter_) {
		std::cout << "load model failed." << std::endl;
		return 10000;
	}

	MNN::ScheduleConfig schedule_config;
	schedule_config.type = MNN_FORWARD_CPU;
	schedule_config.numThread = 4;

	MNN::BackendConfig backend_config;
	backend_config.precision = MNN::BackendConfig::Precision_High;
	backend_config.power = MNN::BackendConfig::Power_High;
	schedule_config.backendConfig = &backend_config;

	onnxSegmentation_sess_ = onnx_interpreter_->createSession(schedule_config);

	// image processer
#if 0
	MNN::CV::Matrix trans;
	trans.setScale(1.0f, 1.0f);
	MNN::CV::ImageProcess::Config img_config;
	img_config.filterType = MNN::CV::BICUBIC;
	::memcpy(img_config.mean, meanVals_, sizeof(meanVals_));
	::memcpy(img_config.normal, normVals_, sizeof(normVals_));

	img_config.sourceFormat = MNN::CV::BGRA;
	img_config.destFormat = MNN::CV::GRAY;
	

	pretreat_data_ = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(img_config));
	pretreat_data_->setMatrix(trans);
#endif

	std::string input_name = "0";//"input.1";//"0";  //293

	input_tensor_ = onnx_interpreter_->getSessionInput(onnxSegmentation_sess_, input_name.c_str());

	nchwTensor = new MNN::Tensor(input_tensor_, MNN::Tensor::CAFFE);  //  add 
	//input_tensor_->copyFromHostTensor(nchwTensor);  //add
	// nchwTensor-host<float>()[x] = ...

	//delete nchwTensor;
	onnx_interpreter_->resizeTensor(input_tensor_, dims_);
	onnx_interpreter_->resizeSession(onnxSegmentation_sess_);

	initialized_ = true;

	std::cout << "end Init." << std::endl;
	return 0;
}

uint8_t* GetImage(const cv::Mat& img_src) {
	uchar* data_ptr = new uchar[img_src.total() * 4];
	cv::Mat img_tmp(img_src.size(), CV_8UC4, data_ptr);
	//cv::Mat img_tmp(img_src.size(), CV_32FC4, data_ptr);
	cv::cvtColor(img_src, img_tmp, cv::COLOR_BGR2RGBA, 4);
	return (uint8_t*)img_tmp.data;
}

uint8_t* GetImage_test(const cv::Mat& img_src,cv::Mat& pre_mask)
{
	uchar* data_ptr = new uchar[img_src.total() * 4];
	uchar* ori_ptr = data_ptr;
	//cv::Mat img_tmp(img_src.size(), CV_8UC4, data_ptr);
	int row = img_src.rows;
	int col = img_src.cols;

#if 1
	
	
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			for (int c = 0; c < 3+1; c++)
			{
				if (c == 3)
				{
					(data_ptr++)[0] = 0;
				}
				else if (c == 0)
				{
					(data_ptr++)[0] = (img_src.at<cv::Vec3b>(i, j)[2]);
				}
				else if (c == 2)
				{
					(data_ptr++)[0] = (img_src.at<cv::Vec3b>(i, j)[0]);
				}
				else
				{
					(data_ptr++)[0] = (img_src.at<cv::Vec3b>(i, j)[c]);
				}
			}


		}
	}
	



#else
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{

				if (c == 0)
				{
					(data_ptr++)[0] = (img_src.at<cv::Vec3b>(i, j)[2]);
				}
				else if (c == 2)
				{
					(data_ptr++)[0] = (img_src.at<cv::Vec3b>(i, j)[0]);
				}
				else
				{
					(data_ptr++)[0] = (img_src.at<cv::Vec3b>(i, j)[c]);
				}



			}
		}
	}
	
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			(data_ptr++)[0] = (pre_mask.at<uchar>(i, j));
		}
	}
	
#endif

	return (uint8_t*)ori_ptr;
}


void fill_data(cv::Mat input_img, cv::Mat pre_mask, float* output, const int index = 0)
{
	cv::Mat dst_img;
	cv::Mat dst_pre;
	cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
	cv::Scalar std = cv::Scalar(0.229, 0.224, 0.225);
	float scale = 0.00392;
	input_img.convertTo(dst_img, CV_32F, scale);
	pre_mask.convertTo(dst_pre, CV_32F, scale);

	dst_img -= mean;
	dst_img /= std;
	int row = dst_img.rows;
	int col = dst_img.cols;
	float* temp = output;
	for (int c = 0; c < 3; c++) {
		
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++) 
			{
			
				//output[c * row * col + i * col + j] = (dst_img.ptr<float>(i, j)[c]);
				temp++[0] = (dst_img.ptr<float>(i, j)[c]);
			}
		}
	}
	if (index % 20 == 0)
	{
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				//output[3 * row * col + i * col + j] = (dst_pre.ptr<float>(i, j)[0]);
				temp++[0] = (dst_pre.ptr<float>(i, j)[0]);
			}
		}
	}



	return;
}
int onnxSegmentation::Detect(cv::Mat img_data, cv::Mat pre_mask,  cv::Mat& out_mask)
{
	std::cout << "start detect." << std::endl;
	if (!initialized_) {
		std::cout << "model uninitialized." << std::endl;
		return 10000;
	}
	if (img_data.empty() || pre_mask.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}

	//int width = img_src.cols;
	//int height = img_src.rows;

	//fill_data(image0, pre_mask, output, index);

	//uint8_t* data_ptr = GetImage(img_data);
	//uint8_t* data_ptr = GetImage_test(img_data, pre_mask);

	//pretreat_data_->convert(data_ptr, inputSize_.width, inputSize_.height, 0, input_tensor_);

	auto change_ptr = nchwTensor->host<float>();

	//nchwTensor - host<float>()[x] = ...
	int index = 0;
	fill_data(img_data, pre_mask, change_ptr, index);


	input_tensor_->copyFromHostTensor(nchwTensor);
	
	//pretreat_data_->convert(data_ptr, 640, 384, 0, , 640, 384, 4, 0, halide_type_of<float>());

	onnx_interpreter_->runSession(onnxSegmentation_sess_);
	std::string output_name ="293";  //"427"; 
	MNN::Tensor* output_tensor = onnx_interpreter_->getSessionOutput(onnxSegmentation_sess_, output_name.c_str());
	//MNN::Tensor* output_tensor = onnx_interpreter_->getSessionOutput(onnxSegmentation_sess_, NULL);

	// copy to host
	MNN::Tensor output_host(output_tensor, output_tensor->getDimensionType());
	output_tensor->copyToHostTensor(&output_host);

	auto output_ptr = output_host.host<float>();

	for (int i = 0; i < output_host.height(); ++i)
	{
		for (int j = 0; j < output_host.width(); ++j)
		{
			if (i == output_host.height() / 2)
			{
				std::cout << output_ptr[j + i * output_host.width()] << " ";
			}
			
			if (output_ptr[j + i * output_host.width()] > 0)
			{
				out_mask.at<uchar>(i, j) = 200;

			}
		}
	
		
	}


	std::cout << "end detect." << std::endl;

	return 0;
}

//
//int main(void)
//{
//	const char* model_path = "D:/pengt/code/inference/segmentation/4channels384_640.mnn";
//	//const char* model_path = "D:/pengt/code/inference/segmentation/3channels384_640.mnn";
//
//	const char* input_image = "D:\\pengt\\segmetation\\test_pic\\1.jpg";
//	const char* out_image = ".\\result.jpg";
//
//	int inputwidth = 640;
//    int inputheight = 384;
//
//	cv::Mat raw_image = cv::imread(input_image);
//	int raw_image_height = raw_image.rows;
//	int raw_image_width = raw_image.cols;
//	cv::Mat image;
//	cv::resize(raw_image, image, cv::Size(inputwidth, inputheight));
//	//cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
//	cv::Mat pre_mask = cv::Mat::zeros(inputheight, inputwidth, CV_8UC1);  //height  width
//
//
//	double timeStart = (double)cv::getTickCount();
//
//	onnxSegmentation my_test;
//	my_test.Init(model_path);
//
//
//	cv::Mat result_mask = cv::Mat::zeros(inputheight, inputwidth, CV_8UC1);
//	my_test.Detect(image, pre_mask, result_mask);
//
//	double nTime = ((double)cv::getTickCount() - timeStart) / cv::getTickFrequency();
//	std::cout << "running time ：" << nTime << "sec\n" << std::endl;
//	cv::imshow("test",result_mask);
//	cv::waitKey(0);
//
//	return 0;
//}