#include"c2_recognized.h"
#include <io.h>
#include<iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <direct.h>
#include<vector>
#include"opencv2/opencv.hpp"
#include <direct.h>

using namespace std;


#define THREAD_NUM   5

#if 1
void get_path_files(const std::string& path, const std::string& ext, std::vector<std::string>& files)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	std::string path_name;
	if ((hFile = _findfirst(path_name.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR)) //子目录
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					get_path_files(path_name.assign(path).append("\\").append(fileinfo.name), ext, files);
				}
			}
			else
			{
				std::string s = path_name.assign(path).append("\\").append(fileinfo.name);

				char fileDrive[_MAX_DRIVE];
				char fileDir[_MAX_DIR];
				char fileName[_MAX_FNAME];
				char fileExt[_MAX_EXT];
				_splitpath_s(s.c_str(), fileDrive, fileDir, fileName, fileExt);

				if (strcmp(fileExt, ext.c_str()) == 0)
				{
					files.push_back(path_name.assign(path).append("\\").append(fileinfo.name));
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
//分割
void split(const string& s, vector<string>& tokens, char delim = '\\')
{
	tokens.clear();
	auto string_find_first_not = [s, delim](size_t pos = 0) -> size_t {
		for (size_t i = pos; i < s.size(); i++) {
			if (s[i] != delim) return i;
		}
		return string::npos;
	};
	size_t lastPos = string_find_first_not(0);
	size_t pos = s.find(delim, lastPos);
	while (lastPos != string::npos) {
		tokens.emplace_back(s.substr(lastPos, pos - lastPos));
		lastPos = string_find_first_not(pos);
		pos = s.find(delim, lastPos);
	}
}
int main()
{
	void* p = nullptr;
	void* p1 = nullptr;

	//D:\\XiAn_Alg_New\\model_path\\Isobar_Loss
	//D:\\XiAn_Alg_New\\model_path\\CurrentCarryingRing_loss
	//std::cout << " 请输入model_path: " << std::endl;
	//std::string model_path;
	//std::cin >> model_path;
	//std::cout << "输入model_path路径为 : " << model_path << std::endl;
	//D:\\XiAn_Alg_New\\model_path\\Isobar_Loss
	//D:\\XiAn_Alg_New\\model_path\\LocationRingReverse
	///////////////////////////卸载测试//////////////////////////
	//cv::Mat img = cv::imread("D:\\西安4C\\缺陷库\\西安二期线路缺陷\\吊弦载流环缺失\\191903926_K12230_47_2_29.jpg", 1);
	//int max_count = 10;
	//int defect_count = 0;
	//DefectResult* defect_results = new DefectResult[10];
	//while (true)
	//{
	//	ResultState Init = InitializeModel(&p, "D:\\XiAn_Alg_New\\model_path\\CurrentCarryingRing_loss");
	//	ImageRecognize(p, img.data, img.cols, img.rows, 3, defect_results, max_count, defect_count);
	//	ResultState release = DisposableModel(&p);
	//}
	///////////////////////////卸载测试//////////////////////////

	//////////////////环境测试/////////////////////////
	//int max_count = 10;
	//int defect_count = 0;
	//DefectResult* defect_results = new DefectResult[10];
	//cv::Mat img = cv::imread("D:\\1.jpg", 1);
	//ResultState Init = InitializeModel(&p, "D:\\CurrentCarryingRing_loss");
	//ImageRecognize(p, img.data, img.cols, img.rows, 3, defect_results, max_count, defect_count);
	//std::cout << "defect_count = " << defect_count << std::endl;
	//////////////////环境测试/////////////////////////
	std::vector<std::string> image_file;
	//std::string read_path;
	//std::cout << " 请输入图片读取路径路径: " << std::endl;
	//std::cin >> read_path;
	//std::cout << "输入图片读取路径为 : " << read_path << std::endl;
	//std::cout << " 请输入保存路径: " << std::endl;
	//std::string save_path;
	//std::cin >> save_path;
	//std::cout << "输入保存路径为 : " << save_path << std::endl;
	
	ResultState Init = InitializeModel(&p, "H:\\2C_CODE\\c2_recognition\\weights\\nest");
	//ResultState Init = InitializeModel(&p, "D:\\XiAn_Alg_New\\model_path\\HangEarFallOut");
	std::string read_path;
	
	read_path = "H:\\测试数据";
	//read_path = "H:\\西安4C\\缺陷库\\西安二期线路缺陷\\平斜腕臂跳线缺失";
	//read_path = "G:\\TRT\\tensorRT_Pro-main\\x64\\Release\\inference";

	std::string save_path = "H:\\RESULTS";
	
	get_path_files(read_path, ".jpg", image_file);
	auto start = std::chrono::system_clock::now();
	for (int j = 0; j < image_file.size(); j++)
	{
		std::cout << (j + 1) << "/" << image_file.size() << "	" << "images_path:" << image_file[j] << std::endl;
		//std::cout << "j = " <<j<< std::endl;
		vector<string> image_name;
		split(image_file[j], image_name);
		//std::cout << image_file[j] << std::endl;
		cv::Mat img = cv::imread(image_file[j], 1);
		DefectResult* defect_results = new DefectResult[10];
		int max_count = 10;
		int defect_count = 0;
		//ImageRecognize(p, img.data, img.cols, img.rows, 3, defect_results, max_count, defect_count);
		auto start = std::chrono::system_clock::now();
		//ImageRecognize(p, img.data, img.cols, img.rows, 3, defect_results, max_count, defect_count);
		const char* save_path_const_char = save_path.c_str();
		const char* img_name = image_name[image_name.size() - 1].c_str();
		//ImageRecognize(p, img.data, img.cols, img.rows, 3, defect_results, max_count, defect_count);
		
		ImageRecognize_Demo(p, img.data, img.cols, img.rows, 3, defect_results, max_count, defect_count, save_path_const_char, img_name);
		
		//如果没有定位到保存位置
		//if (defect_count == 0) {
		//	_mkdir(std::string(save_path + "\\No_object").c_str());
		//	std::string save_file_nobject = save_path + "\\No_object\\" + image_name[image_name.size() - 1];
		//	cv::imwrite(save_file_nobject, img);
		//}
		std::cout << "defect_count = "<< defect_count << std::endl;
		for (int i = 0; i < defect_count; i++)
		{
			cv::Mat image;

			img.copyTo(image);
			std::cout <<"Confidence = " <<defect_results[i].Confidence << std::endl;
			int left, top;
			left = defect_results[i].PositionLeft;
			top = defect_results[i].PositionTop;
			int color_num = i;
			cv::Rect rc;
			rc.x = defect_results[i].PositionLeft;
			rc.y = defect_results[i].PositionTop;
			rc.width = defect_results[i].PositionWidth;
			rc.height = defect_results[i].PositionHeight;
			std::cout << "x = " << rc.x << std::endl;
			std::cout << "Y = " << rc.y << std::endl;
			std::cout << "W = " << rc.width << std::endl;
			std::cout << "H = " << rc.height << std::endl;
						  
			cv::rectangle(image, rc, cv::Scalar(0, 0, 255), 5, 8);

			std::string label = std::string(defect_results[i].Code) + ":" + std::to_string(defect_results[i].Confidence);
			cv::Mat small_pic = img(rc);
			//保存小图
			//if (defect_results[i].Confidence == 0) {
			//	_mkdir(std::string(save_path + "\\Normal").c_str());
			//	std::string save_file_normal = save_path + "\\Normal\\" + image_name[image_name.size() - 1];
			//	cv::imwrite(save_file_normal, small_pic);
			//}
			//else
			//{
			//	_mkdir(std::string(save_path + "\\Defect").c_str());
			//	std::string save_file_defect = save_path + "\\Defect\\" + image_name[image_name.size() - 1];
			//	cv::imwrite(save_file_defect, small_pic);
			//}
			//cv::imwrite("Q:\\loc\\1.jpg", small_pic);
			int baseLine;
			cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			top = std::max(top, labelSize.height);
			//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
			cv::putText(image, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 5, cv::Scalar(0, 0, 255), 5);
			//cv::namedWindow("res", 0);
			//cv::imshow("res", image);
			//cv::waitKey();
			//cv::namedWindow("small_pic", 0);
			//cv::imshow("small_pic", small_pic);
			//cv::waitKey();
			
			std::cout << "缺陷样本" << defect_results[i].Code << std::endl;
			if (defect_results[i].Confidence > 0) {
				_mkdir(std::string(save_path + "\\Draw").c_str());
				std::string save_file_draw = save_path + "\\Draw\\" + image_name[image_name.size() - 1];
				cv::imwrite(save_file_draw, image);
			}
			//std::string save_path2 = "D:\\ZLU_TEST\\测试保存\\" + std::to_string(j) + ".jpg";

		}
		if (defect_count == 0)
		{
			//_mkdir(std::string(save_path + "\\Normal").c_str());
			//std::string save_file_draw = save_path + "\\Normal\\" + image_name[image_name.size() - 1];
			//cv::imwrite(save_file_draw, img);
		}
		delete[]defect_results;
	}
	auto end = std::chrono::system_clock::now();
	std::cout << "time_all_img = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}
#endif



