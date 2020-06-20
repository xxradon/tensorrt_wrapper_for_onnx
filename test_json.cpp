#include "json/json.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include "tensorrt_engine.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace tensorrtInference;

#define GLOBAL_OUTPUT_SIZE 4096
#define LOCAL_OUTPUT_SZIE  90*160*256
#define SCORE_SIZE         720*1280
int main()
{
    std::string jsonFileName = "hfnet_github_graph.json";
    std::string weightsFileName = "hfnet_github_weights.bin";
    // weightsAndGraphParse parser(jsonFileName, weightsFileName);
    // save engine file
#if 0
    tensorrtEngine engine(jsonFileName, weightsFileName);
    // engine.saveEnginePlanFile("hfnet_github.engine");
    engine.saveEnginePlanFile("hfnet_github_fp16.engine");
#else
    std::string bmpFile = "gray_test.bmp";
    cv::Mat colorBmp = cv::imread(bmpFile.c_str());
    cv::Mat grayBmp;
    cv::Mat inputBmp( 721, 1281, CV_8UC1, cv::Scalar(0));
    cv::Rect rect(0, 0, 1280, 720);
    cv::cvtColor(colorBmp, grayBmp, cv::COLOR_BGR2GRAY);
    grayBmp.copyTo(inputBmp(rect));
    cv::Mat inputFloatBmp( 721, 1281, CV_32FC1, cv::Scalar(0));
    inputBmp.convertTo(inputFloatBmp, CV_32F);
    // tensorrtEngine engine("hfnet_github.engine");
    tensorrtEngine engine("hfnet_github_fp16.engine");

    auto bindingNamesIndexMap = engine.getBindingNamesIndexMap();
    float globalDesc[GLOBAL_OUTPUT_SIZE];
    float *localDesc = (float*)malloc(sizeof(float) * LOCAL_OUTPUT_SZIE);
    float *score     = (float*)malloc(sizeof(float) * SCORE_SIZE);
    int   *scoreIndex     = (int*)malloc(sizeof(int) * SCORE_SIZE);
    
    std::vector<void*> data(bindingNamesIndexMap.size());

    data[bindingNamesIndexMap["prefix/image:0"]] = inputFloatBmp.data;
    data[bindingNamesIndexMap["prefix/pred/global_head/l2_normalize:0"]] = &globalDesc[0];
    data[bindingNamesIndexMap["prefix/pred/local_head/descriptor/Mul_1:0"]] = localDesc;
    data[bindingNamesIndexMap["prefix/pred/Reshape:0"]] = score;
    data[bindingNamesIndexMap["prefix/pred/keypoint_extraction/Greater:0"]] = scoreIndex;
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        // engine.doInference(data, 1, false);
        engine.doInference(data, 1, true);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    std::cout << "global desc !!!!!!!!!!!!!!!" << std::endl;
    for(int i = 0; i < 20; i++)
    {
        std::cout << globalDesc[i] << std::endl;
    }
    std::cout << "local desc !!!!!!!!!!!!!!!" << std::endl;
    for(int i = 0; i < 20; i++)
    {
        std::cout << localDesc[i] << std::endl;
    }
    std::cout << "score !!!!!!!!!!!!!!!" << std::endl;
    for(int i = 0; i < 20; i++)
    {
        std::cout << score[i] << std::endl;
    }
    std::cout << "score index !!!!!!!!!!!!!!!" << std::endl;
    std::cout << "score index count is " <<  (scoreIndex[464400 - 1]) << std::endl;
    for(int i = 0; i < 20; i++)
    {
        std::cout << (scoreIndex[i] / 1280) <<  " --- " << (scoreIndex[i] % 1280) << std::endl;
    }    
    free(localDesc);
    free(score);
    free(scoreIndex);
#endif
    
    std::cout << "test weights and graph parser !!!" << std::endl;
}