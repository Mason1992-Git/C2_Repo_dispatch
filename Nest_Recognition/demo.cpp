#include "alg_process.h"
#include"opencv2/opencv.hpp"
#include <fstream>
#include <iostream>
#include <io.h>
#include <string>
#include "c2_recognized.h"

//结构体
//struct Output {
//	int id;//结果类别id
//	float confidence;//结果置信度
//	cv::Rect box;//矩形框
//};
std::vector<std::string> pte_className = { "0", "1" };
int main() {
	//std::cout << "hello world" << std::endl;
	//cv::dnn::Net yolov5_model;
	const char* model_path = "H:\\2C_CODE\\2C_PROJECT\\SDK\\Config_Files\\c2\\nest\\nest_classifier";
	c4_function* Alg_pro = new c4_function();
	int status = Alg_pro->load_model(model_path, 0);
	
	std::string img_path = "H:\\测试数据\\2.jpg";
	std::vector<Output> out;
	cv::Mat img = cv::imread(img_path, 1);
	if (img.empty()) {
		return 0;
	}
	//std::vector<bbox_t> OutResult;
	int result;
	//bool yolo_detect_status=Alg_pro->Detect_yolo(img, pte_className,640,640, Alg_pro->yolov5_Model, out);
	//bool darknet_detect_status = Alg_pro->Detect_darknet(img, Alg_pro->darknet_Model_1, OutResult);
	bool darknet_detect_status = Alg_pro->Classifier_resnet18_dnn(img, Alg_pro->classifier_net_1, result);
	std::cout << "darknet_detect_status = "<< result << std::endl;
	//扩增
	int new_w, new_h, new_x, new_y;





}