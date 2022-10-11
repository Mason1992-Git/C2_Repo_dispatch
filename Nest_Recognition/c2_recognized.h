#pragma once
#pragma once
#ifndef INTERFACE_LIB_API
#define INTERFACE_LIB_API __declspec(dllexport)

#if __cplusplus

typedef enum ResultState {
	//调用成功
	Success = 0,
	//未全局初始化算法环境
	NotGlobalInit = 1,
	//输入的模型路径无效或者文件缺失，或者文件存在问题
	InvalidModel = 2,
	//无效的句柄或者重复初始化 
	ErrorHandler = 3,
	//待检测的图像数据无效(指针或尺寸)
	InvalidImageData = 4,
	//图像通道数不支持
	InvalidImageChannelNum = 5,
	//nvidia显卡不可用
	GpuNotAvailable = 6,
	//nvidia显存不足
	GpuMemInsufficient = 7,
	//未初始化（未调用InitializeModel）或者之前初始化过程中部分模型初始化失败
	NotSetInitMethod = 8,
	//初始化中的算法存在不支持的类型
	InitInvalidMethod = 9,
	//未选择算法类型进行识别（激活）
	NotSetActivateMethod = 10,
	//选择的算法类型中存在未初始化的类型
	ActivateInvalidMethod = 11,
} ResultState;

typedef struct DefectResult {
	//标准缺陷编码-6C标准编码
	char Code[20];
	//置信度，范围0-1
	float Confidence;
	//缺陷小图距离原始图像左边的像素距离
	int PositionLeft;
	//缺陷小图距离原始图像上边的像素距离
	int PositionTop;
	//缺陷小图的像素宽度
	int PositionWidth;
	//缺陷小图的像素高度
	int PositionHeight;
	//缺陷真实长度
	int MeasureHeight;
	//缺陷真实宽度
	int MeasureWidth;
	//缺陷面积
	int MeasureArea;
	//缺陷点集
	int MeasurePointSet;
} DefectResult;

extern "C" {
#endif
	INTERFACE_LIB_API  char* getVersion();

	INTERFACE_LIB_API ResultState InitializeModel(void** handle, const char* model_path);

	INTERFACE_LIB_API ResultState ImageRecognize(void* handle, unsigned char* image_data, int image_width, int image_height, int image_channel, DefectResult* defect_results, int max_count,
		int& defect_count);

	INTERFACE_LIB_API  ResultState ImageRecognize_Demo(void* handle, unsigned char* image_data, int image_width, int image_height, int image_channel, DefectResult* defect_results, int max_count,
		int& defect_count, const char* save_path, const char* img_name);
	
	INTERFACE_LIB_API ResultState ImagePathRecognize(void* handle, const char* image_path, DefectResult* defect_results, int max_count,
		int& defect_count);

	INTERFACE_LIB_API ResultState ImagePathLocated(void* handle, const char* image_path, DefectResult* defect_results, int max_count,
		int& defect_count);

	INTERFACE_LIB_API ResultState DisposableModel(void** handle);
#if __cplusplus
}
#endif
#endif


