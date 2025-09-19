#include <bits/stdc++.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

const std::string model_path = "./models/lama_fp32.onnx";
const std::string image_path = "./images/input_image.jpg";
const std::string mask_path = "./images/dilated_mask.png";
const cv::Size target(512, 512); // if model needs a spesific image size
const int image_idx = 0;
const int mask_idx = 1;
const std::vector<std::string> input_names = {"image", "mask"};
const std::vector<std::string> output_names = {"output"};

static bool SaveOrtOutputAsPng(const Ort::Value &out, const std::string &baseName, size_t index);
static std::vector<int64_t> matShapeToNCHW(const cv::Mat &blob);

int main()
{
    // Read image and mask
    cv::Mat mat_image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat mat_mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
    if (mat_image.empty())
    {
        std::cerr << "image not found: " << image_path << "\n";
        return 2;
    }
    if (mat_mask.empty())
    {
        std::cerr << "mask not found: " << mask_path << "\n";
        return 2;
    }

    // Preprocess inputs
    cv::Mat blob_image = cv::dnn::blobFromImage(
        mat_image, 1.f / 255.f, target, cv::Scalar(), /*swapRB*/ true, /*crop*/ false, CV_32F);
    cv::Mat blob_mask = cv::dnn::blobFromImage(
        mat_mask, 1.f / 255.f, target, cv::Scalar(), /*swapRB*/ false, /*crop*/ false, CV_32F);
    if (!blob_image.isContinuous() || !blob_mask.isContinuous())
        throw std::runtime_error("blob is not continuous");

    float *img_data = reinterpret_cast<float *>(blob_image.data);
    float *msk_data = reinterpret_cast<float *>(blob_mask.data);
    std::vector<int64_t> image_shape = matShapeToNCHW(blob_image); // 1x3xHxW
    std::vector<int64_t> mask_shape = matShapeToNCHW(blob_mask);   // 1x1xHxW

    // Get the first provider
    auto provider = Ort::GetAvailableProviders().front();

    // Setting up ONNX environment
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default");
    Ort::SessionOptions sessionOptions;
    OrtCUDAProviderOptions cuda_options;

    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetIntraOpNumThreads(1);
    // optimization will take time and memory during startup
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try
    {
        // Start an ONNX Runtime session and create CPU memory info for input tensors.
        // model path is const wchar_t*
        const ORTCHAR_T *kModelPath = model_path.c_str();
        Ort::Session session = Ort::Session(env, kModelPath, sessionOptions);

        // Allocate memory for inputs
        Ort::MemoryInfo memory_info{nullptr};
        try
        {
            memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        }
        catch (Ort::Exception oe)
        {
            std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
            return -1;
        }

        // Inputs
        std::vector<const char *> input_names_c;
        for (auto &s : input_names)
            input_names_c.push_back(s.c_str());

        std::vector<Ort::Value> inputs(input_names.size());
        inputs[image_idx] = Ort::Value::CreateTensor<float>(
            mem_info, img_data, (size_t)blob_image.total(), image_shape.data(), image_shape.size());
        inputs[mask_idx] = Ort::Value::CreateTensor<float>(
            mem_info, msk_data, (size_t)blob_mask.total(), mask_shape.data(), mask_shape.size());

        // Outputs
        std::vector<const char *> output_names_c;
        for (auto &s : output_names)
            output_names_c.push_back(s.c_str());

        // Run
        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names_c.data(), inputs.data(), inputs.size(),
            output_names_c.data(), output_names_c.size());

        // Basic Input Summary
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            auto &in = inputs[i];
            auto info = in.GetTensorTypeAndShapeInfo();
            auto shp = info.GetShape();
            std::cout << "Input[" << i << "] \"" << input_names[i] << "\" shape=[";
            for (size_t k = 0; k < shp.size(); ++k)
                std::cout << shp[k] << (k + 1 == shp.size() ? "]\n" : ",");
        }

        // Basic Output Summary
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            auto &out = outputs[i];
            auto info = out.GetTensorTypeAndShapeInfo();
            auto shp = info.GetShape();
            std::cout << "Output[" << i << "] \"" << output_names[i] << "\" shape=[";
            for (size_t k = 0; k < shp.size(); ++k)
                std::cout << shp[k] << (k + 1 == shp.size() ? "]\n" : ",");
        }
        std::cout << "Inference OK\n";

        // Saving outputs as PNG
        for (size_t i = 0; i < outputs.size(); ++i)
            SaveOrtOutputAsPng(outputs[i], "./outputs/output", i);
    }
    catch (Ort::Exception oe)
    {
        std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
        return -1;
    }
    return 0;
}

static std::vector<int64_t> matShapeToNCHW(const cv::Mat &blob)
{
    if (blob.dims != 4)
        throw std::runtime_error("blob must be 4D (NCHW).");
    return {blob.size[0], blob.size[1], blob.size[2], blob.size[3]};
}

// Write single output to PNG
static bool SaveOrtOutputAsPng(const Ort::Value &out, const std::string &baseName, size_t index)
{
    // Take shape
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shp = info.GetShape(); // NCHW expected)
    if (shp.size() != 4 || shp[0] != 1)
    {
        std::cerr << "[SaveOrtOutputAsPng] Expected NCHW with N=1. Got shape:";
        for (size_t i = 0; i < shp.size(); ++i)
            std::cerr << shp[i] << (i + 1 == shp.size() ? "\n" : "x");
        return false;
    }
    const int64_t C = shp[1], H = shp[2], W = shp[3];
    if (!(C == 1 || C == 3))
    {
        std::cerr << "[SaveOrtOutputAsPng] Supports only C=1 or C=3. Got C=" << C << "\n";
        return false;
    }

    // File name: output_HHMMSS_index.png
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << baseName << "_" << std::put_time(&tm, "%H%M%S") << "_" << index << ".png";
    const std::string filename = oss.str();

    cv::Mat image_u8; // (CV_8U, 1 or 3 channel)
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);

    const float *ptr = out.GetTensorData<float>();

    // CHW -> HWC (float)
    if (C == 1)
    {
        cv::Mat ch(H, W, CV_32F, const_cast<float *>(ptr));
        cv::Mat m = ch.clone();
        double minv, maxv;
        cv::minMaxLoc(m, &minv, &maxv);
        if (maxv <= 1.0 + 1e-6 && minv >= 0.0) // between 0 and 1
            m *= 255.0f;

        m.convertTo(image_u8, CV_8U);
    }
    else if (C == 3)
    {
        cv::Mat c0(H, W, CV_32F, const_cast<float *>(ptr + plane * 0));
        cv::Mat c1(H, W, CV_32F, const_cast<float *>(ptr + plane * 1));
        cv::Mat c2(H, W, CV_32F, const_cast<float *>(ptr + plane * 2));
        // Combine planar CHW channels into one interleaved HWC (RGB) image
        std::vector<cv::Mat> ch = {c0, c1, c2};
        cv::Mat img32f;
        cv::merge(ch, img32f);

        double minv, maxv;
        cv::minMaxLoc(img32f.reshape(1), &minv, &maxv);
        if (maxv <= 1.0 + 1e-6 && minv >= 0.0)
            img32f *= 255.0f;

        img32f.convertTo(image_u8, CV_8UC3);
        // switch back to BGR
        cv::cvtColor(image_u8, image_u8, cv::COLOR_RGB2BGR);
    }

    // Write to file
    if (!cv::imwrite(filename, image_u8))
    {
        std::cerr << "[SaveOrtOutputAsPng] Write error: " << filename << "\n";
        return false;
    }
    std::cout << "Saved to: " << filename << "\n";
    return true;
}
