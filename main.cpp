#include<windows.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace Ort;
using namespace std;
using namespace cv;

struct boxs
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
    int gh;
    int gw;
    int stride;
};
HANDLE hcom;
void generate_grids_and_stride(int target_size, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid = target_size / stride;
        for (int g1 = 0; g1 < num_grid; g1++)
        {
            for (int g0 = 0; g0 < num_grid; g0++)
            {
                GridAndStride gs;
                gs.gh = g0;
                gs.gw = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

int main()
{
    vector<Value> input_tensors, output_tensors;        //输入输入创建
    vector<int> strides = { 8, 16, 32};           //网格
    vector<GridAndStride> grid_strides;            //网格
    vector<cv::Rect> boxes;                        //坐标
    vector<int> classIds;                          //类别
    vector<float> confidences;                     //置信度
    vector<int> indices;
    HBITMAP	BitMap, hOld;
    Mat img, img0;
    float conf = 0.4;


    //地址
    //string img_path = "C:/Users/Zzzz/Desktop/Z/000102.png";
    const wchar_t* modelFilepath = L"C:/Users/Zzzz/Desktop/Z/yolox.onnx";  //模型地址   

    //加载图片
    //Mat img = imread(img_path);

    // 环境和选项
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "SuperResolution");    //实例化引擎
    Ort::SessionOptions session_options;    //实例化输入

    //设置图优化级别和线程
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 加载模型并创建会话,以及一些创建操作
    Ort::Session session(env, modelFilepath, session_options);   //加载模型
    Ort::AllocatorWithDefaultOptions allocator;                 //实例化会话
    auto memory_info = MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); //创建CPU
    std::vector<const char*> input_names{ session.GetInputName(0, allocator) };     //语法糖
    std::vector<const char*> output_names{ session.GetOutputName(0, allocator) };
    auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();   //获取输入维度 [1 3 640 640]
    auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape(); //获取输出维度 [1 3549 7]

    // 创建输出维度一样的空矩阵
    Mat output;
    output.create(Size(output_dims[1], output_dims[2]), CV_32F); 
    const int num_class = output_dims[2] - 5;  //类别
    const float img_size = (float)input_dims[2];

    //1.获取屏幕句柄
    HWND hwnd = GetDesktopWindow();
    RECT rect;
    GetWindowRect(hwnd, &rect);
    int cx = (rect.right - rect.left) * 0.5;//800
    int cy = (rect.bottom - rect.top) * 0.5; //450

    //要截取的宽高
    int width = int((640 * 0.5));    //截取的宽 320
    int  height = int((640 * 0.5));     //320
    //根据截取的宽高计算截图原点
    int x = int(cx - (width * 0.5)); //截取的原点x   480
    int y = int(cy - (height * 0.5));//截取的原点y   130

    //2.获取屏幕DC
    HDC hdc = GetWindowDC(hwnd);
    //3.创建兼容DC(内存DC)
    HDC	mfdc = CreateCompatibleDC(hdc);
    //5.创建位图Bitmap对象
    BitMap = CreateCompatibleBitmap(hdc, width, height);
    //6.将位图对象放入内存dc(也可以是绑定)
    SelectObject(mfdc, BitMap);
    //7.创建一个固定维度的空矩阵,
    img0.create(Size(width, height), CV_8UC4);
    int i = 0;
    while (true)
    {
        i++;


        double t1 = getTickCount();
        //get_img
        BitBlt(mfdc, 0, 0, width, height, hdc, x, y, SRCCOPY);
        //将BitBlt的位图信息传入
        GetBitmapBits(BitMap, height * width * 4, img0.data);//位图对象句柄,字节数,需要拷贝到的地方
        cvtColor(img0, img, COLOR_BGRA2BGR);  // 4->3,img0是转换后的图片,用来做输入

        //预处理
        float sx = static_cast<float>(img.cols) / img_size; //比例
        float sy = static_cast<float>(img.rows) / img_size;
        Mat blob = dnn::blobFromImage(img, 1.0, Size(input_dims[2], input_dims[2]), NULL, true, false);      //预处理

        //生成输入输出张量
        input_tensors.clear();
        output_tensors.clear();
        input_tensors.emplace_back(Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size()));
        output_tensors.emplace_back(Value::CreateTensor<float>(memory_info, output.ptr<float>(), output.total(), output_dims.data(), output_dims.size()));

        // 推理   
        session.Run(RunOptions{ nullptr }, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_tensors.data(), output_tensors.size());

        //获取输出数据的指针
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();

        // 生成三个输出层的网格与锚点信息
        grid_strides.clear();
        generate_grids_and_stride(img_size, strides, grid_strides);

        // 解码输出
        boxes.clear();
        classIds.clear();
        confidences.clear();

        for (int anchor_idx = 0; anchor_idx < grid_strides.size(); anchor_idx++)
        {
            const int grid0 = grid_strides[anchor_idx].gh; // H
            const int grid1 = grid_strides[anchor_idx].gw; // W
            const int stride = grid_strides[anchor_idx].stride; // stride
            const int basic_pos = anchor_idx * output_dims[2];       //[,?......] 7

            float x_center = (floatarr[basic_pos + 0] + grid0) * stride * sx;
            float y_center = (floatarr[basic_pos + 1] + grid1) * stride * sy;
            float w = exp(floatarr[basic_pos + 2]) * stride * sx;
            float h = exp(floatarr[basic_pos + 3]) * stride * sy;
            float x0 = x_center - w * 0.5f;     //除法比乘法慢
            float y0 = y_center - h * 0.5f;
            float box_objectness = floatarr[basic_pos + 4];

            for (int class_idx = 0; class_idx < num_class; class_idx++)
            {
                float box_cls_score = floatarr[basic_pos + 5 + class_idx];
                float box_prob = box_objectness * box_cls_score;

                if (box_prob > conf)
                {
                    cv::Rect rect;
                    rect.x = x0;
                    rect.y = y0;
                    rect.width = w;
                    rect.height = h;

                    classIds.push_back(class_idx);
                    confidences.push_back((float)box_prob);
                    boxes.push_back(rect);
                }
            }
        }


        //画框
        indices.clear();
   
        cv::dnn::NMSBoxes(boxes, confidences, 0.1, 0.1, indices);

        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            rectangle(img, boxes[idx], cv::Scalar(0, 255, 0), 2, 8, 0);
        }

        ostringstream ss;
        double time = (getTickCount() - t1) * 1000 / (getTickFrequency());
        ss << "FPS: " << fixed << setprecision(2) << 1000 / time << " ;TIME: " << time << "ms"; //fixed << setprecision(3);控制后面输出的小数点位
        putText(img, ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 0), 2, 8);
        //显示
        //resize(img, img, Size(960, 540));
        imshow("test", img);
        waitKey(1);
    }
    //释放
    DeleteDC(hdc);
    DeleteDC(mfdc);
    DeleteObject(BitMap);
    //system("pause");
    return 0;
}

