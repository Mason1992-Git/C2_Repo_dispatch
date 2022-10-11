#include"c2_recognized.h"
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include"alg_process.h"

std::mutex mtx;
INTERFACE_LIB_API  char* getVersion() {
	return (char*)"V1.0.0.0";
}

INTERFACE_LIB_API  ResultState InitializeModel(void** handle, const char* model_path) {

	//std::cout << "��ʼ��ʼ��ģ��..." << std::endl;
	c4_function* Alg_pro = new c4_function();
	std::string model_P = Alg_pro->UTF8ToGB(model_path);
	std::cout << "����ģ��·��: " << model_P << std::endl;
	*handle = (void*)Alg_pro;
	//handle = &Alg_pro;
	size_t avail(0);//�����Դ�
	size_t total(0);//���Դ�
	int nCudaNums = 0;

	cudaGetDeviceCount(&nCudaNums);//��ȡ�Կ�����
	int cuda_device = -1;
	if (nCudaNums > 0)
	{
		std::cout << "�����Կ�����" << nCudaNums << std::endl;
		cudaDeviceProp prop;
		int MAX_GPU = -1;
		int max_GPU_device = -1;
		for (int i = 0; i < nCudaNums; i++)
		{
			cudaGetDeviceProperties(&prop, i);
			cudaSetDevice(i);
			cudaMemGetInfo(&avail, &total);
			//������������  
			std::cout << "�Կ���ţ�device " << i << " �Կ��ͺţ�" << prop.name << " �Դ�ʣ�ࣺ" << avail / 1024 / 1024 << "MB �Դ�������" << total / 1024 / 1024 << "MB" << std::endl;
			int GPU_MB = avail / 1024 / 1024;
			if (MAX_GPU < GPU_MB)
			{
				max_GPU_device = i;
				MAX_GPU = GPU_MB;
			}
		}
		cudaGetDeviceProperties(&prop, max_GPU_device);
		cudaSetDevice(max_GPU_device);
		cudaMemGetInfo(&avail, &total);
		int GPU_MB = avail / 1024 / 1024;
		std::cout << "�Դ�ʣ������Կ�Ϊ��" << prop.name << " �Դ�ʣ�ࣺ" << GPU_MB << "MB �Դ�������" << total / 1024 / 1024 << "MB" << std::endl;
		if (GPU_MB < 800)
		{
			std::cout << "�Դ���ܲ���" << std::endl;
			return GpuMemInsufficient;
		}
		std::cout << "��ʹ���Կ���" << prop.name << std::endl;
		cuda_device = max_GPU_device;
	}
	else
	{
		std::cout << model_P << "��ʹ��CPU����ģ��" << std::endl;
		//std::cout << model_path << " ��ʼ��ʧ�� " << std::endl;
		//return GpuNotAvailable;
	}

	int status = Alg_pro->load_model(model_path, cuda_device);
	if (status != 0)
	{
		std::cout << model_path << " ��ʼ��ʧ�� " << std::endl;
		return InvalidModel;
	}

	std::cout << *handle << std::endl;

	return Success;
}

INTERFACE_LIB_API  ResultState ImageRecognize(void* handle, unsigned char* image_data, int image_width, int image_height, int image_channel,DefectResult* defect_results, int max_count, int& defect_count)
{
	mtx.lock();

	defect_count = 0;
	std::cout << handle << std::endl;
	std::cout << "����ͼ����: " << image_width << std::endl;
	std::cout << "����ͼ��߶�: " << image_height << std::endl;
	std::cout << "����ͼ��ͨ��: " << image_channel << std::endl;

	if (handle == 0)
	{
		mtx.unlock();
		return NotSetInitMethod;
	}

	if (image_data == 0)
	{
		mtx.unlock();
		return InvalidImageData;
	}
	try
	{
		c4_function* pRun = (c4_function*)handle;
		pRun->defect_detect(image_data, image_width, image_height, image_channel,defect_results, max_count, defect_count);
	}
	catch (...)
	{
		mtx.unlock();
		return NotSetInitMethod;
	}

	mtx.unlock();
	return Success;
}

//DEMO
INTERFACE_LIB_API  ResultState ImageRecognize_Demo(void* handle, unsigned char* image_data, int image_width, int image_height, int image_channel, DefectResult* defect_results, int max_count, int& defect_count, const char* save_path, const char* img_name)
{
	mtx.lock();

	defect_count = 0;
	std::cout << handle << std::endl;
	std::cout << "����ͼ����: " << image_width << std::endl;
	std::cout << "����ͼ��߶�: " << image_height << std::endl;
	std::cout << "����ͼ��ͨ��: " << image_channel << std::endl;

	if (handle == 0)
	{
		mtx.unlock();
		return NotSetInitMethod;
	}

	if (image_data == 0)
	{
		mtx.unlock();
		return InvalidImageData;
	}
	try
	{
		c4_function* pRun = (c4_function*)handle;

		pRun->defect_detect_demo(image_data, image_width, image_height, image_channel, defect_results, max_count, defect_count, save_path, img_name);
	}
	catch (...)
	{
		mtx.unlock();
		return NotSetInitMethod;
	}

	mtx.unlock();
	return Success;
}

INTERFACE_LIB_API ResultState ImagePathRecognize(void* handle, const char* image_path, DefectResult* defect_results, int max_count, int& defect_count)
{
	if (handle == 0 || image_path == 0)
		return NotSetInitMethod;
	try
	{
		//std::cout << handle << std::endl;
		std::cout << image_path << std::endl;
		c4_function* pRun = (c4_function*)handle;
		//pRun->located_4C_Path(image_path, defect_results, max_count, defect_count);
	}
	catch (...)
	{
		return NotSetInitMethod;
	}


	return Success;
}

INTERFACE_LIB_API ResultState ImagePathLocated(void* handle, const char* image_path, DefectResult* defect_results, int max_count,int& defect_count)
{
	if (handle == 0 || image_path == 0)
		return NotSetInitMethod;
	try
	{
		//std::cout << handle << std::endl;
		std::cout << image_path << std::endl;
		c4_function* pRun = (c4_function*)handle;
		//pRun->located_4C_Path_Only(image_path, defect_results, max_count, defect_count);
	}
	catch (...)
	{
		return NotSetInitMethod;
	}


	return Success;
}

INTERFACE_LIB_API  ResultState DisposableModel(void** handle) {
	delete (c4_function*)(*handle);
	*handle = NULL;
	std::cout << "�ͷ�..." << std::endl;
	return Success;
}