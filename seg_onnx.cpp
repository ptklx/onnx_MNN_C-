//
//  segment.cpp
//  MNN
//
//  Created by MNN on 2019/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/AutoTime.hpp>
#include <iostream>
/////////
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
////////


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


using namespace MNN;
using namespace MNN::CV;
using namespace MNN::Express;

//const char model_path [] = "D:/pengt/code/inference/segmentation/deeplabv3_257_mv_gpu.mnn";
//const char* model_path = "D:/pengt/code/inference/segmentation/deeplabv3_257_mv_gpu.mnn";
const char* model_path = "D:/pengt/code/inference/segmentation/4channels384_640.mnn";

const char* input_image = "D:\\pengt\\segmetation\\test_pic\\1.jpg";
const char* out_image = ".\\result.jpg";


#pragma comment(lib, "MNN_debug.lib")


int main(int argc, const char* argv[]) {
    /* if (argc < 4) {
         MNN_PRINT("Usage: ./segment.out model.mnn input.jpg output.jpg\n");
         return 0;
     }*/

     //auto net = Variable::getInputAndOutput(Variable::loadMap(argv[1]));
    auto net = Variable::getInputAndOutput(Variable::loadMap(model_path));
    if (net.first.empty()) {
        MNN_ERROR("Invalid Model\n");
        return 0;
    }
    auto input = net.first.begin()->second;
    auto info = input->getInfo();
    if (nullptr == info) {
        MNN_ERROR("The model don't have init dim\n");
        return 0;
    }
    auto shape = input->getInfo()->dim;
    shape[0] = 1;
    input->resize(shape);
    auto output = net.second.begin()->second;
    if (nullptr == output->getInfo()) {
        MNN_ERROR("Alloc memory or compute size error\n");
        return 0;
    }

    {
        int size_w = 0;
        int size_h = 0;
        int bpp = 0;
        if (info->order == NHWC) {
            bpp = shape[3];
            size_h = shape[1];
            size_w = shape[2];
        }
        else {
            bpp = shape[1];
            size_h = shape[2];
            size_w = shape[3];
        }
        if (bpp == 0)
            bpp = 1;
        if (size_h == 0)
            size_h = 1;
        if (size_w == 0)
            size_w = 1;
        MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

        //auto inputPatch = argv[2];
        auto inputPatch = input_image;
        int width, height, channel;
        auto inputImage = stbi_load(inputPatch, &width, &height, &channel, 4);
        if (nullptr == inputImage) {
            MNN_ERROR("Can't open %s\n", inputPatch);
            return 0;
        }
        MNN_PRINT("origin size: %d, %d\n", width, height);
        CV::Matrix trans;
        // Set scale, from dst scale to src
        trans.setScale((float)(width - 1) / (size_w - 1), (float)(height - 1) / (size_h - 1));
        CV::ImageProcess::Config config;
        config.filterType = CV::BILINEAR;
        //        float mean[3]     = {103.94f, 116.78f, 123.68f};
        //        float normals[3] = {0.017f, 0.017f, 0.017f};
        float mean[4] = { 127.5f, 127.5f, 127.5f,125 };
        float normals[4] = { 0.00785f, 0.00785f, 0.00785f,1 };

        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(normals));
        config.sourceFormat = CV::BGRA;
        config.destFormat = CV::GRAY;

        std::shared_ptr<CV::ImageProcess> pretreat(CV::ImageProcess::create(config));
        pretreat->setMatrix(trans);
        pretreat->convert((uint8_t*)inputImage, width, height, 0, input->writeMap<float>(), size_w, size_h, 4, 0, halide_type_of<float>());
        stbi_image_free(inputImage);

        input->unMap();
    }
    {
        //auto originOrder = output->getInfo()->order;
        output = _Convert(output, NCHW);
        //output = _Softmax(output, -1);
        auto outputInfo = output->getInfo();
        auto width = outputInfo->dim[3];
        auto height = outputInfo->dim[2];
        auto channel = outputInfo->dim[1];
        std::shared_ptr<Tensor> wrapTensor(CV::ImageProcess::createImageTensor<uint8_t>(width, height, 4, nullptr));
        MNN_PRINT("Mask: w=%d, h=%d, c=%d\n", width, height, channel);
        auto output_ptr = output->readMap<float>();

        //MNN::Tensor output_host(output_tensor, output_tensor->getDimensionType());
        //output_tensor->copyToHostTensor(&output_host);

        //auto output_ptr = output_host.host<float>();

        cv::Mat result_mask = cv::Mat::zeros(height, width, CV_8UC1);;
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                if (i == height / 2)
                {
                    std::cout << output_ptr[j + i * width] << " ";
                }

                if (output_ptr[j + i * width] > 0)
                {
                    result_mask.at<uchar>(i, j) = 200;

                }
            }


        }
        cv::imshow("test", result_mask);
        cv::waitKey(0);
       /* for (int y = 0; y < height; ++y) {
            auto rgbaY = wrapTensor->host<uint8_t>() + 4 * y * width;
            auto sourceY = outputHostPtr + y * width * channel;
            for (int x = 0; x < width; ++x) {
                auto sourceX = sourceY + channel * x;
                int index = 0;
                float maxValue = sourceX[0];
                auto rgba = rgbaY + 4 * x;
                for (int c = 1; c < channel; ++c) {
                    if (sourceX[c] > maxValue) {
                        index = c;
                        maxValue = sourceX[c];
                    }
                }
                rgba[0] = 255;
                rgba[2] = 0;
                rgba[1] = 0;
                rgba[3] = 255;
                if (15 == index) {
                    rgba[2] = 255;
                    rgba[3] = 0;
                }
            }
        }*/
        output->unMap();
        //stbi_write_png(out_image, width, height, 4, wrapTensor->host<uint8_t>(), 4 * width);
    }
    return 0;
}
