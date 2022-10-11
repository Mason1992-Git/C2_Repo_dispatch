#pragma once
#pragma once
#ifndef INTERFACE_LIB_API
#define INTERFACE_LIB_API __declspec(dllexport)

#if __cplusplus

typedef enum ResultState {
	//���óɹ�
	Success = 0,
	//δȫ�ֳ�ʼ���㷨����
	NotGlobalInit = 1,
	//�����ģ��·����Ч�����ļ�ȱʧ�������ļ���������
	InvalidModel = 2,
	//��Ч�ľ�������ظ���ʼ�� 
	ErrorHandler = 3,
	//������ͼ��������Ч(ָ���ߴ�)
	InvalidImageData = 4,
	//ͼ��ͨ������֧��
	InvalidImageChannelNum = 5,
	//nvidia�Կ�������
	GpuNotAvailable = 6,
	//nvidia�Դ治��
	GpuMemInsufficient = 7,
	//δ��ʼ����δ����InitializeModel������֮ǰ��ʼ�������в���ģ�ͳ�ʼ��ʧ��
	NotSetInitMethod = 8,
	//��ʼ���е��㷨���ڲ�֧�ֵ�����
	InitInvalidMethod = 9,
	//δѡ���㷨���ͽ���ʶ�𣨼��
	NotSetActivateMethod = 10,
	//ѡ����㷨�����д���δ��ʼ��������
	ActivateInvalidMethod = 11,
} ResultState;

typedef struct DefectResult {
	//��׼ȱ�ݱ���-6C��׼����
	char Code[20];
	//���Ŷȣ���Χ0-1
	float Confidence;
	//ȱ��Сͼ����ԭʼͼ����ߵ����ؾ���
	int PositionLeft;
	//ȱ��Сͼ����ԭʼͼ���ϱߵ����ؾ���
	int PositionTop;
	//ȱ��Сͼ�����ؿ��
	int PositionWidth;
	//ȱ��Сͼ�����ظ߶�
	int PositionHeight;
	//ȱ����ʵ����
	int MeasureHeight;
	//ȱ����ʵ���
	int MeasureWidth;
	//ȱ�����
	int MeasureArea;
	//ȱ�ݵ㼯
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


