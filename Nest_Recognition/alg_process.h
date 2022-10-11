#pragma once
#define OPENCV
#include<vector>
#include <io.h>
#include<iostream>
#include <fstream>
#include <sstream>
#include <direct.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "c2_recognized.h"
#include "HalconCpp.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// darknet
#include "darknet/yolo_v2_class.hpp"

using namespace HalconCpp;
using namespace std;
//yolov5输出结构体
struct Output {
	int id;//结果类别id
	float confidence;//结果置信度
	cv::Rect box;//矩形框
};

class c4_function {
public:
	c4_function();
	~c4_function();
	//yolov5模型，一级定位
	cv::dnn::Net yolov5_Model_1;
	cv::dnn::Net yolov5_Model_2;
	cv::dnn::Net yolov5_Model_3;
	cv::dnn::Net yolov5_Model_4;
	cv::dnn::Net yolov5_Model_5;
	//dnn 分类
	cv::dnn::Net classifier_net_1;
	//u2net模型，分割
	//cv::dnn::Net u2net_Model;
	//u2net模型，libtorch
	torch::jit::script::Module u2net_Model_libtorch;
	//darknet(yolov3)模型
	Detector* darknet_Model_1 = nullptr;
	Detector* darknet_Model_2 = nullptr;
	//读取txt文件并转为整型
	void read_txt_convert(std::string path, int& config_flag);
	//夜间区分隧道内外函数
	bool classifier_tunnel(cv::Mat& cv_image, int& tunnel_flag);
	//Halcon转换函数
	bool Mat2HObject(cv::Mat& cv_image, HalconCpp::HObject& h_image);
	//Halcon转换函数
	bool HObject2MatImg(HalconCpp::HObject& Hobj, cv::Mat& matImg);
	//畸形定位去除函数
	void invalid_coord(int scale_thresh,int x, int y, int w, int h, int img_width, int img_height,int& out_flag);
	//mat-SoftMax函数
	bool dnn_softmax(const cv::Mat & src, cv::Mat & dst);
	//读取模型
	bool readModel(cv::dnn::Net& net, std::string& netPath, int isCuda, int cuda_device);
	//读取libtorch模型
	bool readModel_libtorch(torch::jit::script::Module& net, std::string& netPath);
	//读取darknet模型
	Detector* readModel_darknet(std::string& cfgPath, std::string& weightPath, bool& status_flag);
	//UTF8TGBK
	std::string UTF8ToGB(const char* str);
	//读取文件函数
	int read_txt(std::string path, std::vector<std::vector<std::string>>& s);
	//分割模型降维函数
	void find_max_id(std::vector<float>scores, int& max_id);
	//分割模型降维
	void find_class_id(cv::Mat src4D, int net_width, int net_height, std::vector<cv::Mat>& dstMat);
	//字符分割
	void spiltStr(std::string str, const std::string& split, std::vector<std::string>& strlist);
	//模型加载函数
	int load_model(const char* model_path, int cuda_device);
	//YOLOV5检测函数
	bool Detect_yolo(cv::Mat& SrcImg, std::vector<std::string> className, const int netWidth, const int netHeight, cv::dnn::Net& net, std::vector<Output>& output);
	//U2NET分割函数
	bool Detect_U2net(cv::Mat& SrcImg, cv::dnn::Net& net);
	//U2NET分割函数,libtorch
	bool Detect_U2net_libtorch(cv::Mat& SrcImg, torch::jit::script::Module& net, cv::Mat& OutImg);
	//resnet18分类函数,libtorch
	bool Classifier_resnet18_libtorch(cv::Mat& SrcImg, torch::jit::script::Module& net, float result);
	//resnet18分类函数,dnn
	bool Classifier_resnet18_dnn(cv::Mat& SrcImg, cv::dnn::Net& net, int &result);
	//darknet检测函数
	bool Detect_darknet(cv::Mat& SrcImg, Detector* net, std::vector<bbox_t>& OutResult);
	//缺陷检测函数
	bool defect_detect(unsigned char* image_data, int image_width, int image_height, int image_channel, DefectResult* defect_results, int max_count, int& defect_count);
	//缺陷检测函数demo
	bool defect_detect_demo(unsigned char* image_data, int image_width, int image_height, int image_channel, DefectResult* defect_results, int max_count, int& defect_count, std::string save_path, std::string image_name);

private:
	//定义相机号
	std::vector<std::string>camera_id;
	//多相机综合判断配置标志位
	int config_flag = 0;
	//stride
	const float netStride[3] = { 8.0, 16.0, 32.0 };
	//anchors
	const float netAnchors[3][6] = { { 10.0, 13.0, 16.0, 30.0, 33.0, 23.0 },{ 30.0, 61.0, 62.0, 45.0, 59.0, 119.0 },{ 116.0, 90.0, 156.0, 198.0, 373.0, 326.0 } };
	//阈值设定
	float nmsThreshold = 0.45;
	float boxThreshold = 0.25;
	float classThreshold = 0.25;
	float confidenceshold = 0.25;
	//类名
	std::vector<std::string> pte_className = { "0", "1", "2","3", "4", "5" ,"6", "7", "8", "9", "10", "11", "12","13","14","15","16","17","18","19","20" };
	//网络模型输入大小
	int netWidth = 640;
	int netHeight = 640;
	//分类网络输入大小
	int classifier_netsize = 224;
	//类别数
	int classes_1 = 1;
	int classes_2 = 1;
	int classes_3 = 1;
	int classes_4 = 1;
	int classes_5 = 1;
	//0代表cpu 1代表gpu 2代表纯配置 3代表真假相加
	int if_cuda = 1;
	//classname
	std::vector<std::string> className_1;
	std::vector<std::string> className_2;
	std::vector<std::string> className_3;
	std::vector<std::string> className_4;
	std::vector<std::string> className_5;
	//设置CPU或GPU
	torch::DeviceType device_type;
	
	

};

