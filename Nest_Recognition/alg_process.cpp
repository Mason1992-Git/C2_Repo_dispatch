#include "alg_process.h"

c4_function::c4_function()
{
}

c4_function::~c4_function()
{
}
//读取文件
int c4_function::read_txt(std::string path, std::vector<std::vector<std::string>>& s)
{
	std::ifstream inFile(path, std::ios::in);
	std::string lineStr;
	//vector<vector<string>> strArray;
	//vector<vector<float>>points;
	while (std::getline(inFile, lineStr))
	{
		std::stringstream ss(lineStr);
		std::string str;
		//vector<string> lineArray;
		std::vector<std::string>num;
		// 按照逗号分隔
		while (std::getline(ss, str, ' '))
		{
			if (str == "")
				continue;
			num.push_back(str);
		}
		s.push_back(num);
	}
	int i = 0;
	if (s.size() < 1)
	{
		return 1;
	}
	return 0;
}
//UTF8TGBK
std::string c4_function::UTF8ToGB(const char* str)
{

	std::string result;
	WCHAR* strSrc;
	LPSTR szRes;

	//获得临时变量的大小
	int i = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
	strSrc = new WCHAR[i + 1];
	MultiByteToWideChar(CP_UTF8, 0, str, -1, strSrc, i);

	//获得临时变量的大小
	i = WideCharToMultiByte(CP_ACP, 0, strSrc, -1, NULL, 0, NULL, NULL);
	szRes = new CHAR[i + 1];
	WideCharToMultiByte(CP_ACP, 0, strSrc, -1, szRes, i, NULL, NULL);
	result = szRes;
	delete[]strSrc;
	delete[]szRes;
	return result;
}
//读取txt文件并转为整型
void c4_function::read_txt_convert(std::string path, int& config_flag) {
	std::ifstream c2_indata1(path.c_str());
	std::string c2_linedata1;
	if (c2_indata1.is_open()) {
		while (getline(c2_indata1, c2_linedata1)) {
			int ret = sscanf(c2_linedata1.c_str(), "%d", &config_flag);
		}
	}
}
//夜间区分隧道内外函数
bool c4_function::classifier_tunnel(cv::Mat& cv_image,int& tunnel_flag) {
	//判断隧道和外景
	if (cv_image.channels() == 3) {
		cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2GRAY);
	}
	int channels[] = { 0 };
	int bins = 256;
	cv::Mat hist;
	int hist_size[] = { bins };
	float range[] = { 0,256 };
	const float*ranges[] = { range };
	cv::calcHist(&cv_image, 1, 0, cv::Mat(), hist, 1, hist_size, ranges);
	//double max_val;
	//minMaxLoc(hist, 0, &max_val);
	float temp_max = 0;
	int temp_id;
	for (int i = 0; i < bins; i++)
	{

		float bin_val = hist.at<float>(i);//图像的灰度频率表
		if (bin_val > temp_max) {
			temp_max = bin_val;
			temp_id = i;
		}
		//std::cout <<"id = " << i <<"  "<<"bin_val = " << bin_val << std::endl;
	}
	//std::cout << "max_val = " << temp_max << std::endl;
	//std::cout << "temp_id = " << temp_id << std::endl;
	if (temp_id < 45) {
		//外景返回1
		tunnel_flag = 1;
	}
	else
	{
		//隧道返回0
		tunnel_flag = 0;

	}
	return true;
}
//Halcon转换函数
bool c4_function::Mat2HObject(cv::Mat& cv_image, HalconCpp::HObject& h_image)
{
	if (cv_image.empty()) {
		return false;
	}

	//2. 三通道
	if (cv_image.channels() == 3) {
		std::vector<cv::Mat> BgrVector(3);
		cv::split(cv_image, BgrVector);
		size_t size = static_cast<size_t>(cv_image.rows * cv_image.cols);
		uchar* BlueData = new uchar[size];
		uchar* GreenData = new uchar[size];
		uchar* RedData = new uchar[size];
		memcpy(BlueData, BgrVector[0].data, size);
		memcpy(GreenData, BgrVector[1].data, size);
		memcpy(RedData, BgrVector[2].data, size);
		HalconCpp::GenImage3(&h_image,
			"byte",
			cv_image.cols,
			cv_image.rows,
			reinterpret_cast<Hlong>(RedData),
			reinterpret_cast<Hlong>(GreenData),
			reinterpret_cast<Hlong>(BlueData));
		delete[] RedData;
		delete[] GreenData;
		delete[] BlueData;
	}
	else if (cv_image.channels() == 1) {
		uchar* GrayData = new uchar[static_cast<size_t>(cv_image.rows * cv_image.cols)];
		memcpy(GrayData, cv_image.data, static_cast<size_t>(cv_image.rows * cv_image.cols));
		HalconCpp::GenImage1(&h_image, "byte", cv_image.cols, cv_image.rows, reinterpret_cast<Hlong>(GrayData));
		delete[] GrayData;
	}

	return true;
}
//Halcon转换函数
bool c4_function::HObject2MatImg(HalconCpp::HObject& Hobj, cv::Mat& matImg)
{
	HalconCpp::HTuple htCh;
	HalconCpp::HString cType;

	HalconCpp::ConvertImageType(Hobj, &Hobj, "byte");
	HalconCpp::CountChannels(Hobj, &htCh);
	Hlong wid = 0;
	Hlong hgt = 0;
	if (htCh[0].I() == 1)
	{
		HalconCpp::HImage hImg(Hobj);
		void* ptr = hImg.GetImagePointer1(&cType, &wid, &hgt);
		int W = wid;
		int H = hgt;
		matImg = cv::Mat::zeros(H, W, CV_8UC1);
		unsigned char* pdata = static_cast<unsigned char*>(ptr);
		memcpy(matImg.data, pdata, W * H);
	}
	else if (htCh[0].I() == 3)
	{
		void* Rptr;
		void* Gptr;
		void* Bptr;
		HalconCpp::HImage hImg(Hobj);
		hImg.GetImagePointer3(&Rptr, &Gptr, &Bptr, &cType, &wid, &hgt);
		int W = wid;
		int H = hgt;
		matImg = cv::Mat::zeros(H, W, CV_8UC3);
		std::vector<cv::Mat> VecM(3);
		VecM[0].create(H, W, CV_8UC1);
		VecM[1].create(H, W, CV_8UC1);
		VecM[2].create(H, W, CV_8UC1);
		unsigned char* R = (unsigned char*)Rptr;
		unsigned char* G = (unsigned char*)Gptr;
		unsigned char* B = (unsigned char*)Bptr;
		memcpy(VecM[2].data, R, W * H);
		memcpy(VecM[1].data, G, W * H);
		memcpy(VecM[0].data, B, W * H);
		cv::merge(VecM, matImg);
	}

	return true;
}
//字符分割
void c4_function::spiltStr(std::string str, const std::string& split, std::vector<std::string>& strlist)
{
	strlist.clear();
	if (str == "")
		return;
	std::string strs = str + split;
	size_t pos = strs.find(split);
	int steps = split.size();

	while (pos != strs.npos)
	{
		//substr 复制字符串，起始位置，复制字符数目
		std::string temp = strs.substr(0, pos);
		strlist.push_back(temp);
		strs = strs.substr(pos + steps, strs.size());
		pos = strs.find(split);
	}

}
//畸形定位去除函数
void c4_function::invalid_coord(int scale_thresh,int x, int y, int w, int h, int img_width, int img_height,int& out_flag) {
	//去除靠近图像边缘图片
	if (x < 3 || (img_width -x - w )< 3 || y < 3||(img_height-y-h)<3) {
		out_flag = 1;
	}
	int max_scale = std::max(w / h, h / w);
	//去除长宽比大于n的图片
	if (max_scale > scale_thresh) {
		out_flag = 1;
	}
}
//分割模型降维函数
void c4_function::find_max_id(std::vector<float>scores, int& max_id)
{
	float max_score = -FLT_MAX;
	for (int i = 0; i < scores.size(); i++)
	{
		if (scores[i] > max_score)
		{
			max_id = i;
			max_score = scores[i];
		}
	}
}
//分割模型降维
void c4_function::find_class_id(cv::Mat src4D, int net_width, int net_height, std::vector<cv::Mat>& dstMat)
{
	for (int i = 0; i < dstMat.size(); i++)
	{
		dstMat[i] = cv::Mat::zeros(cv::Size(net_width, net_height), CV_8UC1);
		//dstMat[i] = cv::Mat::zeros(cv::Size(net_width, net_height), CV_8UC3);
	}
	float* data = (float*)src4D.data;
	//float box_score = data[0];
	//std::cout << "pdata = " << box_score << std::endl;
	for (int h = 0; h < net_height; h++)
	{
		for (int w = 0; w < net_width; w++)
		{
			std::vector<float>scores;
			int max_id = -1;
			//std::cout << "dstMat.size() = " << dstMat.size() << std::endl;
			for (int c = 0; c < dstMat.size(); c++)
			{
				//dstMat[0].at<uchar>(h, w) = int((*(float*)src4D.data * 255));
				scores.push_back(*((float*)src4D.data + c * net_width * net_height + h * net_width + w));
				//std::cout << "src4D.data = " << *((float*)src4D.data + c * net_width * net_height + h * net_width + w) << std::endl;
			}
			find_max_id(scores, max_id);
			dstMat[max_id].at<uchar>(h, w) = 255;
		}
	}
}
//mat-SoftMax函数
bool c4_function::dnn_softmax(const cv::Mat & src, cv::Mat & dst)
{
	float max = 0.0;
	float sum = 0.0;

	max = *max_element(src.begin<float>(), src.end<float>());
	cv::exp((src - max), dst);
	sum = cv::sum(dst)[0];
	dst /= sum;

	return true;
}
//读取模型
bool c4_function::readModel(cv::dnn::Net& net, std::string& netPath, int isCuda, int cuda_device) {
	try {
		net = cv::dnn::readNetFromONNX(netPath);
	}
	catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
		return false;
	}
	//cuda
	if (isCuda && cuda_device >= 0) {
		cv::cuda::setDevice(cuda_device);
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	//cpu
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}

//读取libtorch模型(默认加载到0卡)
bool c4_function::readModel_libtorch(torch::jit::script::Module& net, std::string& netPath) {

	int	iGpuCount = 0;
	cudaGetDeviceCount(&iGpuCount);
	if (iGpuCount < 1)
	{
		std::cout << "未找到显卡" << std::endl;
	}
	else
	{
		cudaDeviceProp kCudaProp;
		for (int i = 0; i < iGpuCount; i++)
		{
			cudaGetDeviceProperties(&kCudaProp, i);
			std::cout << "显卡算力:" << kCudaProp.major << "." << kCudaProp.minor << std::endl;
			if (kCudaProp.major > 3) {
				if (torch::cuda::is_available())
				{

					device_type = torch::kCUDA;
					try {
						net = torch::jit::load(netPath, device_type);
					}
					catch (const c10::Error& e) {
						std::cerr << "error loading the GPU model\n" << e.what();
						return false;
					}
				}
				else
				{
					device_type = torch::kCPU;
					try {
						net = torch::jit::load(netPath, device_type);
					}
					catch (const c10::Error& e) {
						std::cerr << "error loading the CPU model\n" << e.what();
						return false;
					}
				}
			}
			else
			{
				device_type = torch::kCPU;
				try {
					net = torch::jit::load(netPath, device_type);
				}
				catch (const c10::Error& e) {
					std::cerr << "error loading the CPU model\n" << e.what();
					return false;
				}
			}
		}
	}


	if (device_type == torch::kCUDA)
	{
		net.to(torch::kHalf);
	}
	net.eval();
	return true;
}
//读取darknet模型
Detector* c4_function::readModel_darknet(std::string& cfgPath, std::string& weightPath,bool& status_flag) {
	try {
		Detector* net = new Detector(cfgPath, weightPath);
		status_flag = true;
		return net;
	}
	catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
		status_flag = false;
	}
}

//加载模型
int c4_function::load_model(const char* model_path, int cuda_device)
{
	std::string m_path = UTF8ToGB(model_path);

	//读取多相机综合判断配置
	std::string config_flag_path = m_path + "\\config.txt";
	//int config_flag;
	read_txt_convert(config_flag_path, config_flag);
	netWidth = 640;
	netHeight = 640;
	classes_1 = 5;
	classes_2 = 1;
	classes_3 = 1;
	classes_4 = 1;
	classes_5 = 1;
	//classes_3 = 2;
	if_cuda = 1;
	//一级定位模型,分割2C图像
	std::string yolo_path_1 = m_path + "\\" + "2C-partsplit.onnx";
	bool status_m1 = readModel(yolov5_Model_1, yolo_path_1, if_cuda, cuda_device);
	//水泥支柱鸟窝检测
	std::string yolo_path_2 = m_path + "\\" + "2C-nest-pillar0.onnx";
	bool status_m2 = readModel(yolov5_Model_2, yolo_path_2, if_cuda, cuda_device);
	//钢架支柱鸟窝检测
	std::string yolo_path_3 = m_path + "\\" + "2C-nest-pillar1.onnx";
	bool status_m3 = readModel(yolov5_Model_3, yolo_path_3, if_cuda, cuda_device);
	//大限界鸟窝检测
	std::string yolo_path_4 = m_path + "\\" + "2C-nest-daxianjie.onnx";
	bool status_m4 = readModel(yolov5_Model_4, yolo_path_4, if_cuda, cuda_device);
	//横梁鸟窝检测模型
	std::string yolo_path_5 = m_path + "\\" + "2C-nest-crossbeam.onnx";
	bool status_m5 = readModel(yolov5_Model_5, yolo_path_5, if_cuda, cuda_device);
	//鸟窝分类模型
	std::string classifier_path_1 = m_path + "\\" + "2C-nest-classifier.onnx";
	bool status_m6 = readModel(classifier_net_1, classifier_path_1, if_cuda, cuda_device);

	className_1.clear();
	className_2.clear();
	className_3.clear();
	className_4.clear();
	className_5.clear();
	for (int i = 0; i < classes_1; i++)
	{
		className_1.push_back(pte_className[i]);
	}
	for (int i = 0; i < classes_2; i++)
	{
		className_2.push_back(pte_className[i]);
	}
	for (int i = 0; i < classes_3; i++)
	{
		className_3.push_back(pte_className[i]);
	}
	for (int i = 0; i < classes_4; i++)
	{
		className_4.push_back(pte_className[i]);
	}
	for (int i = 0; i < classes_5; i++)
	{
		className_5.push_back(pte_className[i]);
	}
	
	if (status_m1 == false || status_m2 == false || status_m3 == false || 
		status_m4 == false || status_m5 == false|| status_m6 ==false)
	{
		std::cout << "model_error" << std::endl;
		return 2;
	}
	return 0;
}
//int c4_function::load_model(const char* model_path, int cuda_device)
//{
//	std::string m_path = UTF8ToGB(model_path);
//	//读取多相机综合判断配置
//	std::string config_flag_path = m_path + "\\config.txt";
//	//int config_flag;
//	read_txt_convert(config_flag_path, config_flag);
//
//	if_cuda = 1;
//	//分类模型
//	std::string yolo_path_1 = m_path + "\\" + "calssifier.onnx";
//	bool status_m1 = readModel(classifier_net_1, yolo_path_1, if_cuda, cuda_device);
//	if (status_m1 == false)
//	{
//		std::cout << "model_error" << std::endl;
//		return 2;
//	}
//	return 0;
//}
//YOLOV5检测函数

bool c4_function::Detect_yolo(cv::Mat& SrcImg, std::vector<std::string> className, const int netWidth, const int netHeight, cv::dnn::Net& net, std::vector<Output>& output) {
	cv::Mat blob;
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	cv::Mat netInputImg = SrcImg.clone();
	if (netInputImg.channels() == 1)
	{
		cv::cvtColor(netInputImg, netInputImg, cv::COLOR_GRAY2BGR);
	}

	cv::Mat resizeImg = cv::Mat::zeros(maxLen, maxLen, CV_8UC3);
	//std::cout << "size = " << resizeImg.cols << resizeImg.rows << std::endl;
	netInputImg.copyTo(resizeImg(cv::Rect(0, 0, col, row)));
	netInputImg = resizeImg;
	//std::cout << "size = " << netInputImg.cols << netInputImg.rows << netInputImg.channels()<< std::endl;


	cv::dnn::blobFromImage(netInputImg, blob, 1.0 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(), true, false);
	net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	//vector<string> outputLayerName{"345","403", "461","output" };
	//net.forward(netOutputImg, outputLayerName[3]); //获取output的输出
	net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
	//std::cout << "falg " << std::endl;
	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	float ratio_h = (float)netInputImg.rows / netHeight;
	float ratio_w = (float)netInputImg.cols / netWidth;
	int net_width = className.size() + 5;  //输出的网络宽度是类别数+5
	float* pdata = (float*)netOutputImg[0].data;
	for (int stride = 0; stride < 3; stride++) {    //stride
		int grid_x = (int)(netWidth / netStride[stride]);
		int grid_y = (int)(netHeight / netStride[stride]);
		for (int anchor = 0; anchor < 3; anchor++) { //anchors
			const float anchor_w = netAnchors[stride][anchor * 2];
			const float anchor_h = netAnchors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_y; i++) {
				for (int j = 0; j < grid_y; j++) {
					float box_score = pdata[4];//获取每一行的box框中含有某个物体的概率
					if (box_score > boxThreshold) {
						//为了使用minMaxLoc(),将85长度数组变成Mat对象
						cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);
						cv::Point classIdPoint;
						double max_class_socre;
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre = (float)max_class_socre;
						if ((max_class_socre * box_score) > confidenceshold) {
							//rect [x,y,w,h]
							float x = pdata[0];  //x
							float y = pdata[1];   //y
							float w = pdata[2];   //w
							float h = pdata[3];  //h
							int left = (x - 0.5 * w) * ratio_w;
							int top = (y - 0.5 * h) * ratio_h;
							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre * box_score);
							boxes.push_back(cv::Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
						}
					}
					pdata += net_width;//指针移到下一行
				}
			}
		}
	}
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, classThreshold, nmsThreshold, nms_result);
	for (int i = 0; i < nms_result.size(); i++) {
		int idx = nms_result[i];
		Output result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
	}

	if (output.size())
	{
		return true;
	}
	else
		return false;


}//这个括号是最末尾的，包括下面添加之后

 //U2NET分割函数(dnn)

bool c4_function::Detect_U2net(cv::Mat& SrcImg,cv::dnn::Net& net) {
	cv::Mat blob;
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	cv::Mat netInputImg = SrcImg.clone();
	if (netInputImg.channels() == 1)
	{
		cv::cvtColor(netInputImg, netInputImg, cv::COLOR_GRAY2RGB);
	}
	cv::cvtColor(netInputImg, netInputImg, cv::COLOR_BGR2RGB);
	int u2net_netwidth = 256;
	int u2net_netheight = 256;
	cv::dnn::blobFromImage(netInputImg, blob, 1.0 / 255.0, cv::Size(u2net_netwidth, u2net_netheight), cv::Scalar(0, 0, 0), true, false);
	net.setInput(blob);

	cv::Mat outputs = net.forward();
	//std::cout << "outputs = " << outputs.size << std::endl;
	std::vector<cv::Mat>outputImgs;
	outputImgs.resize(2);
	find_class_id(outputs, u2net_netwidth, u2net_netheight, outputImgs);
	//std::cout << "outputImgs = " << outputImgs.size() << std::endl;
	//std::cout << "outputImgs[0] = " << outputImgs[0].dims << std::endl;
	//std::cout << "outputImgs[0] = " << outputImgs[0].cols << std::endl;
	//std::cout << "outputImgs[0] = " << outputImgs[0].rows << std::endl;
	//std::cout << "outputs = " << outputs.rows << std::endl;
	//std::cout << "channels = " << outputs.dims << std::endl;
	//float* pdata = (float*)outputs.data;
	//std::cout << "out = " << *pdata << std::endl;
	//保存
	std::string save_file_path = "D:\\XiAn_Alg_New\\result\\3.jpg";
	cv::imwrite(save_file_path, outputImgs[0]);
	return true;
}

//U2NET分割函数(libtorch)
bool c4_function::Detect_U2net_libtorch(cv::Mat& SrcImg, torch::jit::script::Module& net, cv::Mat& OutImg) {
	cv::Mat netInputImg = SrcImg.clone();
	if (netInputImg.channels() == 1) {
		cv::cvtColor(netInputImg, netInputImg, cv::COLOR_GRAY2RGB);
	}
	cv::cvtColor(netInputImg, netInputImg, cv::COLOR_BGR2RGB);
	cv::Mat seg_img;
	double ratio = 256. / std::max(netInputImg.rows, netInputImg.cols);
	cv::resize(netInputImg, seg_img, cv::Size(std::max(int(ratio * SrcImg.cols), 5), std::max(int(ratio * SrcImg.rows), 5)));
	cv::Mat black_img = cv::Mat::zeros(256, 256, CV_8UC3);
	cv::Mat roi_aera = black_img(cv::Rect(0, 0, seg_img.cols, seg_img.rows));
	seg_img.copyTo(roi_aera);
	auto s2_imgTensor = torch::from_blob(black_img.data, { 1, black_img.rows, black_img.cols, black_img.channels() }, torch::kByte).toType(torch::kFloat).to(device_type);
	if (device_type == torch::kCUDA)
	{
		s2_imgTensor = s2_imgTensor.to(torch::kHalf);
	}
	s2_imgTensor = s2_imgTensor.permute({ 0, 3, 1, 2 }).contiguous();
	s2_imgTensor /= 255.;
	std::vector<torch::jit::IValue> s2_second_inputs;
	s2_second_inputs.emplace_back(s2_imgTensor);
	auto seg_output = net.forward(s2_second_inputs);
	auto seg_preds = seg_output.toTuple()->elements()[0].toTensor();
	torch::Tensor result_pred = seg_preds[0][0].mul(255.0).toType(torch::kByte).to(torch::kCPU);
	cv::Mat img_C1;
	img_C1.create(cv::Size(256, 256), CV_8UC1);
	memcpy(img_C1.data, result_pred.data_ptr(), result_pred.numel() * sizeof(torch::kByte));
	cv::Mat seg_thresh;
	cv::threshold(img_C1, seg_thresh, 200, 250, cv::THRESH_BINARY);
	OutImg = seg_thresh;
	return true;

}
//darknet检测函数
bool c4_function::Detect_darknet(cv::Mat& SrcImg, Detector* net, std::vector<bbox_t>& OutResult ) {
	try {
		if (SrcImg.channels() == 3) {
			cv::cvtColor(SrcImg, SrcImg, cv::COLOR_BGR2GRAY);
		}
		int image_height = SrcImg.rows;
		int image_width = SrcImg.cols;
		cv::Mat netInputImg = SrcImg.clone();

		OutResult = net->detect(SrcImg);
	}
	catch (const std::exception &e) {
		std::cout << e.what() << std::endl;
		return false;
	}

	return true;
}
//resnet18分类函数(二分类)
bool c4_function::Classifier_resnet18_libtorch(cv::Mat& SrcImg, torch::jit::script::Module& net,float result) {
	try {
		cv::Mat netInputImg = SrcImg.clone();
		cv::Mat dst;
		double ratio = 64. / std::max(netInputImg.rows, netInputImg.cols);
		cv::resize(netInputImg, dst, cv::Size(std::max(int(ratio * netInputImg.cols), 5), std::max(int(ratio * netInputImg.rows), 5)));
		//创建256x256黑色背景图
		cv::Mat black_img = cv::Mat(64, 64, CV_8UC1, cv::Scalar(0));
		//设置ROI区域
		cv::Mat roi_aera = black_img(cv::Rect(0, 0, dst.cols, dst.rows));
		//复制
		dst.copyTo(roi_aera, dst);
		torch::Tensor tensor_image = torch::from_blob(black_img.data, { 1, black_img.rows,black_img.cols,1 }, torch::kByte).toType(torch::kFloat);
		//////////////////等比缩代码///////////////////////////////
		//cv::resize(input_2, dst, cv::Size(64, 64));
		//torch::Tensor tensor_image = torch::from_blob(dst.data, { 1, dst.rows,dst.cols,1 }, torch::kByte).toType(torch::kFloat);
		//tensor_image = tensor_image.permute({ 0, 3, 1, 2 });
		//tensor_image /= 255.;
		tensor_image = tensor_image.permute({ 0, 3, 1, 2 });
		tensor_image /= 255.;

		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(tensor_image);
		//infer推理过程
		auto rst = net.forward(inputs).toTensor();
		torch::Tensor rst1 = torch::squeeze(torch::squeeze(torch::squeeze(rst, 2), 2), 0);
		torch::Tensor rst2 = torch::nn::functional::softmax(rst1, 0);

		float rst_float = torch::argmax(rst2, 0).item().toFloat();
		result = rst_float;
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return false;
	}
	return true;
	
}

//resnet18分类函数(dnn)
bool c4_function::Classifier_resnet18_dnn(cv::Mat& SrcImg, cv::dnn::Net& net, int &result) {
	try {
		int col = SrcImg.cols;
		int row = SrcImg.rows;
		int maxLen = MAX(col, row);
		cv::Mat netInputImg = SrcImg.clone();
		if (netInputImg.channels() == 1)
		{
			cv::cvtColor(netInputImg, netInputImg, cv::COLOR_GRAY2BGR);
		}
		cv::Mat resizeImg = cv::Mat::zeros(maxLen, maxLen, CV_8UC3);
		//std::cout << "size = " << resizeImg.cols << resizeImg.rows << std::endl;
		netInputImg.copyTo(resizeImg(cv::Rect(0, 0, col, row)));
		netInputImg = resizeImg;
		cv::Mat input = cv::dnn::blobFromImage(netInputImg,1.0/255, cv::Size(classifier_netsize, classifier_netsize), cv::Scalar(), false, false);
		net.setInput(input);
		cv::Mat predicted = net.forward();
		//std::cout << "result = " << predicted << std::endl;
		cv::Mat soft_pre;
		dnn_softmax(predicted, soft_pre);
		double minValue,maxValue;    // 最大值，最小值
		cv::Point  minIdx, maxIdx;    // 最小值坐标，最大值坐标 
		//int minIdx, maxIdx;
		cv::minMaxLoc(soft_pre, &minValue, &maxValue, &minIdx, &maxIdx);
		//std::cout << "minValue = " << minValue << std::endl;
		//std::cout << "maxValue = " << maxValue << std::endl;
		//std::cout << "minIdx = " << minIdx.x << std::endl;
		//std::cout << "maxIdx = " << maxIdx.x << std::endl;
		result = maxIdx.x;

	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return false;
	}
	return true;

}
//缺陷检测函数
bool c4_function::defect_detect(unsigned char* image_data, int image_width, int image_height, int image_channel, DefectResult* defect_results, int max_count, int& defect_count)
{
	try {
		cv::Mat img;
		if (image_channel == 1)
		{
			img = cv::Mat(image_height, image_width, CV_8UC1, image_data);
			cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
		}
		else if (image_channel == 3)
		{
			img = cv::Mat(image_height, image_width, CV_8UC3, image_data);
			//cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		}
		//定位环一级定位
		//cv::imshow("1", img);
		//cv::waitKey(0);
		std::vector<bbox_t> locationring = darknet_Model_1->detect(img);
		if (locationring.size() < 1)
		{
			std::cout << "not located" << std::endl;
		}
		cv::Rect s1_region;
		bool region_flag = false;
		cv::Mat input_1;
		int out_index = 0;
		for (auto &tbbox : locationring) {
			//处理id为0的部分（即定位环）且考虑双杆
			if (tbbox.obj_id == 0) {
				//s1_region = cv::Rect(tbbox.x, tbbox.y, min(tbbox.w, image_width - tbbox.x - 1), min(tbbox.h, image_height - tbbox.y - 1));
				s1_region = cv::Rect(tbbox.x, tbbox.y, std::min(int(tbbox.w), int(image_width - tbbox.x - 1)), std::min(int(tbbox.h), int(image_height - tbbox.y - 1)));
				img(s1_region).copyTo(input_1);
				//去除靠边或畸形图
				//一级定位畸形框去除
				//长宽比超过阈值
				int scale_thresh = 3;
				int invalid_flag = 0;
				invalid_coord(scale_thresh, tbbox.x, tbbox.y, tbbox.w, tbbox.h, image_width, image_height, invalid_flag);
				if (invalid_flag == 1) {
					continue;
				}
				//二级定位
				std::vector<bbox_t> location_ring = darknet_Model_2->detect(input_1);
				if (location_ring.size() < 1) {
					continue;
				}
				for (auto &tbbox2 : location_ring) {
					//缺陷
					if (tbbox2.obj_id == 1) {
						std::cout << "WARNING:LocationRing Reverse..." << std::endl;
						std::string code = "002003002002";
						strcpy(defect_results[out_index].Code, code.c_str());
						defect_results[out_index].Confidence = 1.0;
						defect_results[out_index].PositionLeft = tbbox.x;
						defect_results[out_index].PositionTop = tbbox.y;
						defect_results[out_index].PositionWidth = tbbox.w;
						defect_results[out_index].PositionHeight = tbbox.h;
						out_index++;
						continue;
					}
				}
			}
		}

		defect_count = out_index > max_count ? max_count : out_index;
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return false;
	}

	return true;
}
//缺陷检测函数demo
bool c4_function::defect_detect_demo(unsigned char* image_data, int image_width, int image_height, int image_channel, DefectResult* defect_results, int max_count, int& defect_count, std::string save_path, std::string image_name)
{
	try {
		cv::Mat img;
		if (image_channel == 1)
		{
			img = cv::Mat(image_height, image_width, CV_8UC1, image_data);
			cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
		}
		else if (image_channel == 3)
		{
			img = cv::Mat(image_height, image_width, CV_8UC3, image_data);
		}
		//一级定位分割支柱、横跨（梁）、大限界、腕臂区域
		std::vector<Output>out_1;
		Detect_yolo(img, className_1, netWidth, netHeight, yolov5_Model_1, out_1);
		if (out_1.size() < 1)
		{
			//无目标保存
			_mkdir(std::string(save_path + "\\Part_split").c_str());
			_mkdir(std::string(save_path + "\\Part_split" + "\\No_object").c_str());
			std::string save_file_No_object = save_path + "\\Part_split" + "\\No_object\\" + image_name;
			cv::imwrite(save_file_No_object, img);
			std::cout << "not located" << std::endl;
		}
		int new_w_1, new_h_1, new_x_1, new_y_1;
		int out_index = 0;
		for (int i = 0; i < out_1.size() && i < max_count; i++) {
			new_x_1 = std::max(out_1[i].box.x,0);
			new_y_1 = std::max(out_1[i].box.y,0);
			new_w_1 = out_1[i].box.width;
			new_h_1 = out_1[i].box.height;
			//一级定位取图
			cv::Rect partsplit_region = cv::Rect(new_x_1, new_y_1, std::min(new_w_1, image_width - new_x_1 - 1), std::min(new_h_1, image_height - new_y_1 - 1));
			cv::Mat target_aera0;
			img(partsplit_region).copyTo(target_aera0);
			
			if (out_1[i].id == 0) {
				//一级定位保存 0
				_mkdir(std::string(save_path + "\\Part_split").c_str());
				_mkdir(std::string(save_path + "\\Part_split" + "\\0").c_str());
				std::string img_name = std::to_string(i) + "_" + image_name;
				std::string save_file_normal = save_path + "\\Part_split" + "\\0\\" + img_name;
				cv::imwrite(save_file_normal, target_aera0);
				//水泥支柱鸟窝检测
				std::vector<Output>out_nest_pillar0;
				Detect_yolo(target_aera0, className_2, netWidth, netHeight, yolov5_Model_2, out_nest_pillar0);
				if (out_nest_pillar0.size() < 1)
				{
					//水泥支柱鸟窝检测无目标保存
					_mkdir(std::string(save_path + "\\nest_pillar0").c_str());
					_mkdir(std::string(save_path + "\\nest_pillar0" + "\\No_object").c_str());
					std::string img_name = std::to_string(i) + "_" + image_name;
					std::string save_file_No_object = save_path + "\\nest_pillar0" + "\\No_object\\" + img_name;
					cv::imwrite(save_file_No_object, target_aera0);
					continue;
				}
				int new_w_2, new_h_2, new_x_2, new_y_2;
				for (int j = 0; j < out_nest_pillar0.size() && j < max_count; j++) {
					new_x_2 = std::max(out_nest_pillar0[j].box.x,0);
					new_y_2 = std::max(out_nest_pillar0[j].box.y,0);
					new_w_2 = out_nest_pillar0[j].box.width;
					new_h_2 = out_nest_pillar0[j].box.height;
					//水泥支柱鸟窝定位取图
					cv::Rect pillar0_region = cv::Rect(new_x_2, new_y_2, std::min(new_w_2, target_aera0.cols - new_x_2 - 1), std::min(new_h_2, target_aera0.rows - new_y_2 - 1));
					cv::Mat nest_pillar0;
					target_aera0(pillar0_region).copyTo(nest_pillar0);
					//水泥支柱鸟窝检测保存
					_mkdir(std::string(save_path + "\\nest_pillar0").c_str());
					_mkdir(std::string(save_path + "\\nest_pillar0" + "\\yolo_nest").c_str());	
					std::string img_name = std::to_string(j) + "_" + image_name;
					std::string save_file_No_object = save_path + "\\nest_pillar0" + "\\yolo_nest\\" + img_name;
					cv::imwrite(save_file_No_object, nest_pillar0);
					//去除畸形结果
					int scale_thresh = 3;
					int invalid_flag = 0;
					//std::cout << "new_x = " << new_x << std::endl;
					//invalid_coord(scale_thresh, new_x_2, new_y_2, new_w_2, new_h_2, target_aera0.cols, target_aera0.rows, invalid_flag);
					//if (invalid_flag == 1) {
					//	continue;
					//}
					//鸟窝检测结果分类
					int nest_classifier_result;
					Classifier_resnet18_dnn(nest_pillar0, classifier_net_1, nest_classifier_result);
					//结果输出
					if (nest_classifier_result == 1) {
						//保存缺陷
						_mkdir(std::string(save_path + "\\nest_pillar0").c_str());
						_mkdir(std::string(save_path + "\\nest_pillar0" + "\\defect").c_str());
						std::string img_name = std::to_string(j) + "_" + image_name;
						std::string save_file_No_object = save_path + "\\nest_pillar0" + "\\defect\\" + img_name;
						cv::imwrite(save_file_No_object, nest_pillar0);
						//支柱鸟窝
						std::string code = "002002009039";
						strcpy(defect_results[out_index].Code, code.c_str());
						//defect_results[out_index].Confidence = out[i].confidence;
						defect_results[out_index].Confidence = 1.0;
						defect_results[out_index].PositionLeft = new_x_1 + new_x_2-10;
						defect_results[out_index].PositionTop = new_y_1 + new_y_2-10;
						defect_results[out_index].PositionWidth = new_w_2;
						defect_results[out_index].PositionHeight = new_h_2;
						out_index++;
					}

				}
			}
			else if (out_1[i].id == 1) {
				//一级定位保存 1
				_mkdir(std::string(save_path + "\\Part_split").c_str());
				_mkdir(std::string(save_path + "\\Part_split" + "\\1").c_str());
				std::string img_name = std::to_string(i) + "_" + image_name;
				std::string save_file_normal = save_path + "\\Part_split" + "\\1\\" + img_name;
				cv::imwrite(save_file_normal, target_aera0);
				//钢架支柱鸟窝检测
				std::vector<Output>out_nest_pillar1;
				Detect_yolo(target_aera0, className_3, netWidth, netHeight, yolov5_Model_3, out_nest_pillar1);
				if (out_nest_pillar1.size() < 1)
				{
					//钢架支柱鸟窝检测无目标保存
					_mkdir(std::string(save_path + "\\nest_pillar1").c_str());
					_mkdir(std::string(save_path + "\\nest_pillar1" + "\\No_object").c_str());
					std::string img_name = std::to_string(i) + "_" + image_name;
					std::string save_file_No_object = save_path + "\\nest_pillar1" + "\\No_object\\" + img_name;
					cv::imwrite(save_file_No_object, target_aera0);
					continue;
				}
				int new_w_2, new_h_2, new_x_2, new_y_2;
				for (int j = 0; j < out_nest_pillar1.size() && j < max_count; j++) {
					new_x_2 = std::max(out_nest_pillar1[j].box.x,0);
					new_y_2 = std::max(out_nest_pillar1[j].box.y,0);
					new_w_2 = out_nest_pillar1[j].box.width;
					new_h_2 = out_nest_pillar1[j].box.height;
					//std::cout << "new_x_2 = "<< new_x_2 <<std::endl;
					//std::cout << "new_y_2 = " << new_y_2 << std::endl;
					//std::cout << "new_w_2 = " << new_w_2 << std::endl;
					//std::cout << "new_h_2 = " << new_h_2 << std::endl;
					//水泥支柱鸟窝定位取图
					cv::Rect pillar1_region = cv::Rect(new_x_2, new_y_2, std::min(new_w_2, target_aera0.cols - new_x_2 - 1), std::min(new_h_2, target_aera0.rows - new_y_2 - 1));
					cv::Mat nest_pillar1;
					target_aera0(pillar1_region).copyTo(nest_pillar1);
					//水泥支柱鸟窝检测保存
					_mkdir(std::string(save_path + "\\nest_pillar1").c_str());
					_mkdir(std::string(save_path + "\\nest_pillar1" + "\\yolo_nest").c_str());
					std::string img_name = std::to_string(j) + "_" + image_name;
					std::string save_file_No_object = save_path + "\\nest_pillar1" + "\\yolo_nest\\" + img_name;
					cv::imwrite(save_file_No_object, nest_pillar1);
					//去除畸形结果
					int scale_thresh = 3;
					int invalid_flag = 0;
					//std::cout << "new_x = " << new_x << std::endl;
					//invalid_coord(scale_thresh, new_x_2, new_y_2, new_w_2, new_h_2, target_aera0.cols, target_aera0.rows, invalid_flag);
					//if (invalid_flag == 1) {
					//	continue;
					//}
					//鸟窝检测结果分类
					int nest_classifier_result;
					Classifier_resnet18_dnn(nest_pillar1, classifier_net_1, nest_classifier_result);
					//结果输出
					if (nest_classifier_result == 1) {
						//保存缺陷
						_mkdir(std::string(save_path + "\\nest_pillar1").c_str());
						_mkdir(std::string(save_path + "\\nest_pillar1" + "\\defect").c_str());
						std::string img_name = std::to_string(j) + "_" + image_name;
						std::string save_file_No_object = save_path + "\\nest_pillar1" + "\\defect\\" + img_name;
						cv::imwrite(save_file_No_object, nest_pillar1);
						//支柱鸟窝
						std::string code = "002002009039";
						strcpy(defect_results[out_index].Code, code.c_str());
						//defect_results[out_index].Confidence = out[i].confidence;
						defect_results[out_index].Confidence = 1.0;
						defect_results[out_index].PositionLeft = new_x_1 + new_x_2-10;
						defect_results[out_index].PositionTop = new_y_1 + new_y_2-10;
						defect_results[out_index].PositionWidth = new_w_2;
						defect_results[out_index].PositionHeight = new_h_2;
						out_index++;
					}

				}
			}
			else if (out_1[i].id == 2) {
				continue;
				////一级定位保存 2
				//_mkdir(std::string(save_path + "\\Part_split").c_str());
				//_mkdir(std::string(save_path + "\\Part_split" + "\\2").c_str());
				//std::string img_name = std::to_string(i) + "_" + image_name;
				//std::string save_file_normal = save_path + "\\Part_split" + "\\2\\" + img_name;
				//cv::imwrite(save_file_normal, target_aera0);
				////腕臂区域鸟窝检测
				//std::vector<Output>out_nest_wrist;
				//Detect_yolo(target_aera0, className_4, netWidth, netHeight, yolov5_Model_4, out_nest_wrist);
				//if (out_nest_wrist.size() < 1)
				//{
				//	//钢架支柱鸟窝检测无目标保存
				//	_mkdir(std::string(save_path + "\\nest_wrist").c_str());
				//	_mkdir(std::string(save_path + "\\nest_wrist" + "\\No_object").c_str());
				//	std::string img_name = std::to_string(i) + "_" + image_name;
				//	std::string save_file_No_object = save_path + "\\nest_wrist" + "\\No_object\\" + img_name;
				//	cv::imwrite(save_file_No_object, target_aera0);
				//	continue;
				//}
				//int new_w_2, new_h_2, new_x_2, new_y_2;
				//for (int j = 0; j < out_nest_wrist.size() && j < max_count; j++) {
				//	new_x_2 = out_nest_wrist[j].box.x;
				//	new_y_2 = out_nest_wrist[j].box.y;
				//	new_w_2 = out_nest_wrist[j].box.width;
				//	new_h_2 = out_nest_wrist[j].box.height;
				//	//水泥支柱鸟窝定位取图
				//	cv::Rect nest_wrist_region = cv::Rect(new_x_2, new_y_2, std::min(new_w_2, image_width - new_x_2 - 1), std::min(new_h_2, image_height - new_y_2 - 1));
				//	cv::Mat nest_wrist;
				//	target_aera0(nest_wrist_region).copyTo(nest_wrist);
				//	//水泥支柱鸟窝检测保存
				//	_mkdir(std::string(save_path + "\\nest_wrist").c_str());
				//	_mkdir(std::string(save_path + "\\nest_wrist" + "\\yolo_nest").c_str());
				//	std::string img_name = std::to_string(j) + "_" + image_name;
				//	std::string save_file_No_object = save_path + "\\nest_wrist" + "\\yolo_nest\\" + img_name;
				//	cv::imwrite(save_file_No_object, nest_wrist);
				//	//去除畸形结果
				//	int scale_thresh = 3;
				//	int invalid_flag = 0;
				//	//std::cout << "new_x = " << new_x << std::endl;
				//	invalid_coord(scale_thresh, new_x_2, new_y_2, new_w_2, new_h_2, target_aera0.cols, target_aera0.rows, invalid_flag);
				//	if (invalid_flag == 1) {
				//		continue;
				//	}
				//	//鸟窝检测结果分类
				//	int nest_classifier_result;
				//	Classifier_resnet18_dnn(nest_wrist, classifier_net_1, nest_classifier_result);
				//	//结果输出
				//	if (nest_classifier_result == 1) {
				//		//保存缺陷
				//		_mkdir(std::string(save_path + "\\nest_wrist").c_str());
				//		_mkdir(std::string(save_path + "\\nest_wrist" + "\\defect").c_str());
				//		std::string img_name = std::to_string(j) + "_" + image_name;
				//		std::string save_file_No_object = save_path + "\\nest_wrist" + "\\defect\\" + img_name;
				//		cv::imwrite(save_file_No_object, nest_wrist);
				//		//支柱鸟窝
				//		std::string code = "002002009039";
				//		strcpy(defect_results[out_index].Code, code.c_str());
				//		//defect_results[out_index].Confidence = out[i].confidence;
				//		defect_results[out_index].Confidence = 1.0;
				//		defect_results[out_index].PositionLeft = new_x_1 + new_x_2;
				//		defect_results[out_index].PositionTop = new_y_1 + new_y_2;
				//		defect_results[out_index].PositionWidth = new_w_2;
				//		defect_results[out_index].PositionHeight = new_h_2;
				//		out_index++;
				//	}
				//}

			}
			else if (out_1[i].id == 3) {
				//一级定位保存 3
				_mkdir(std::string(save_path + "\\Part_split").c_str());
				_mkdir(std::string(save_path + "\\Part_split" + "\\3").c_str());
				std::string img_name = std::to_string(i) + "_" + image_name;
				std::string save_file_normal = save_path + "\\Part_split" + "\\3\\" + img_name;
				cv::imwrite(save_file_normal, target_aera0);
				//大限界鸟窝检测
				std::vector<Output>out_nest_daxianjie;
				Detect_yolo(target_aera0, className_4, netWidth, netHeight, yolov5_Model_4, out_nest_daxianjie);
				if (out_nest_daxianjie.size() < 1)
				{
					//钢架支柱鸟窝检测无目标保存
					_mkdir(std::string(save_path + "\\nest_daxianjie").c_str());
					_mkdir(std::string(save_path + "\\nest_daxianjie" + "\\No_object").c_str());
					std::string img_name = std::to_string(i) + "_" + image_name;
					std::string save_file_No_object = save_path + "\\nest_daxianjie" + "\\No_object\\" + img_name;
					cv::imwrite(save_file_No_object, target_aera0);
					continue;
				}
				int new_w_2, new_h_2, new_x_2, new_y_2;
				for (int j = 0; j < out_nest_daxianjie.size() && j < max_count; j++) {
					new_x_2 = std::max(out_nest_daxianjie[j].box.x,0);
					new_y_2 = std::max(out_nest_daxianjie[j].box.y,0);
					new_w_2 = out_nest_daxianjie[j].box.width;
					new_h_2 = out_nest_daxianjie[j].box.height;
					//水泥支柱鸟窝定位取图
					cv::Rect nest_daxianjie_region = cv::Rect(new_x_2, new_y_2, std::min(new_w_2, target_aera0.cols - new_x_2 - 1), std::min(new_h_2, target_aera0.rows - new_y_2 - 1));
					cv::Mat nest_daxianjie;
					target_aera0(nest_daxianjie_region).copyTo(nest_daxianjie);
					//水泥支柱鸟窝检测保存
					_mkdir(std::string(save_path + "\\nest_daxianjie").c_str());
					_mkdir(std::string(save_path + "\\nest_daxianjie" + "\\yolo_nest").c_str());
					std::string img_name = std::to_string(j) + "_" + image_name;
					std::string save_file_No_object = save_path + "\\nest_daxianjie" + "\\yolo_nest\\" + img_name;
					cv::imwrite(save_file_No_object, nest_daxianjie);
					//去除畸形结果
					int scale_thresh = 3;
					int invalid_flag = 0;
					//std::cout << "new_x = " << new_x << std::endl;
					//invalid_coord(scale_thresh, new_x_2, new_y_2, new_w_2, new_h_2, target_aera0.cols, target_aera0.rows, invalid_flag);
					//if (invalid_flag == 1) {
					//	continue;
					//}
					//鸟窝检测结果分类
					int nest_classifier_result;
					Classifier_resnet18_dnn(nest_daxianjie, classifier_net_1, nest_classifier_result);
					//结果输出
					if (nest_classifier_result == 1) {
						//保存缺陷
						_mkdir(std::string(save_path + "\\nest_daxianjie").c_str());
						_mkdir(std::string(save_path + "\\nest_daxianjie" + "\\defect").c_str());
						std::string img_name = std::to_string(j) + "_" + image_name;
						std::string save_file_No_object = save_path + "\\nest_daxianjie" + "\\defect\\" + img_name;
						cv::imwrite(save_file_No_object, nest_daxianjie);
						//支柱鸟窝
						std::string code = "002002009039";
						strcpy(defect_results[out_index].Code, code.c_str());
						//defect_results[out_index].Confidence = out[i].confidence;
						defect_results[out_index].Confidence = 1.0;
						defect_results[out_index].PositionLeft = new_x_1 + new_x_2-10;
						defect_results[out_index].PositionTop = new_y_1 + new_y_2-10;
						defect_results[out_index].PositionWidth = new_w_2;
						defect_results[out_index].PositionHeight = new_h_2;
						out_index++;
					}

				}
			}
			else if (out_1[i].id == 4) {
				//一级定位保存 4
				_mkdir(std::string(save_path + "\\Part_split").c_str());
				_mkdir(std::string(save_path + "\\Part_split" + "\\4").c_str());
				std::string img_name = std::to_string(i) + "_" + image_name;
				std::string save_file_normal = save_path + "\\Part_split" + "\\4\\" + img_name;
				cv::imwrite(save_file_normal, target_aera0);
				//硬横跨鸟窝检测
				std::vector<Output>out_nest_crossbeam;
				Detect_yolo(target_aera0, className_5, netWidth, netHeight, yolov5_Model_5, out_nest_crossbeam);
				if (out_nest_crossbeam.size() < 1)
				{
					//钢架支柱鸟窝检测无目标保存
					_mkdir(std::string(save_path + "\\nest_crossbeam").c_str());
					_mkdir(std::string(save_path + "\\nest_crossbeam" + "\\No_object").c_str());
					std::string img_name = std::to_string(i) + "_" + image_name;
					std::string save_file_No_object = save_path + "\\nest_crossbeam" + "\\No_object\\" + img_name;
					cv::imwrite(save_file_No_object, target_aera0);
					continue;
				}
				int new_w_2, new_h_2, new_x_2, new_y_2;
				for (int j = 0; j < out_nest_crossbeam.size() && j < max_count; j++) {
					new_x_2 = std::max(out_nest_crossbeam[j].box.x,0);
					new_y_2 = std::max(out_nest_crossbeam[j].box.y,0);
					new_w_2 = out_nest_crossbeam[j].box.width;
					new_h_2 = out_nest_crossbeam[j].box.height;
					//水泥支柱鸟窝定位取图
					cv::Rect nest_crossbeam_region = cv::Rect(new_x_2, new_y_2, std::min(new_w_2, target_aera0.cols - new_x_2 - 1), std::min(new_h_2, target_aera0.rows - new_y_2 - 1));
					cv::Mat nest_crossbeam;
					target_aera0(nest_crossbeam_region).copyTo(nest_crossbeam);
					//水泥支柱鸟窝检测保存
					_mkdir(std::string(save_path + "\\nest_crossbeam").c_str());
					_mkdir(std::string(save_path + "\\nest_crossbeam" + "\\yolo_nest").c_str());
					std::string img_name = std::to_string(j) + "_" + image_name;
					std::string save_file_No_object = save_path + "\\nest_crossbeam" + "\\yolo_nest\\" + img_name;
					cv::imwrite(save_file_No_object, nest_crossbeam);
					//去除畸形结果
					int scale_thresh = 3;
					int invalid_flag = 0;
					//std::cout << "new_x = " << new_x << std::endl;
					//invalid_coord(scale_thresh, new_x_2, new_y_2, new_w_2, new_h_2, target_aera0.cols, target_aera0.rows, invalid_flag);
					//if (invalid_flag == 1) {
					//	continue;
					//}
					//鸟窝检测结果分类
					int nest_classifier_result;
					Classifier_resnet18_dnn(nest_crossbeam, classifier_net_1, nest_classifier_result);
					//结果输出
					if (nest_classifier_result == 1) {
						//保存缺陷
						_mkdir(std::string(save_path + "\\nest_crossbeam").c_str());
						_mkdir(std::string(save_path + "\\nest_crossbeam" + "\\defect").c_str());
						std::string img_name = std::to_string(j) + "_" + image_name;
						std::string save_file_No_object = save_path + "\\nest_crossbeam" + "\\defect\\" + img_name;
						cv::imwrite(save_file_No_object, nest_crossbeam);
						//支柱鸟窝
						std::string code = "002002009039";
						strcpy(defect_results[out_index].Code, code.c_str());
						//defect_results[out_index].Confidence = out[i].confidence;
						defect_results[out_index].Confidence = 1.0;
						defect_results[out_index].PositionLeft = new_x_1 + new_x_2-10;
						defect_results[out_index].PositionTop = new_y_1 + new_y_2-10;
						defect_results[out_index].PositionWidth = new_w_2;
						defect_results[out_index].PositionHeight = new_h_2;
						out_index++;
					}

				}
			}
		}
		defect_count = out_index > max_count ? max_count : out_index;
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return false;
	}

	return true;
}
