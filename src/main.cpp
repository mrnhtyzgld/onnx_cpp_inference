#include "InferenceRunner.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>

bool save_pngs(const std::vector<cv::Mat> &mats,
               const std::string &dir,
               const std::string &base)
{
    bool all_ok = true;
    for (size_t i = 0; i < mats.size(); ++i)
    {
        const cv::Mat &img = mats[i];
        if (img.empty())
        {
            std::cerr << "[save_pngs] empty mat at index " << i << "\n";
            all_ok = false;
            continue;
        }

        // filename: <dir>/<base>_<i>.png
        std::string filename = dir;
        if (!dir.empty() && dir.back() != '/' && dir.back() != '\\')
            filename += '/';
        filename += base + "_" + std::to_string(i) + ".png";

        if (!cv::imwrite(filename, img))
        {
            std::cerr << "[save_pngs] write error: " << filename << "\n";
            all_ok = false;
        }
        else
        {
            std::cout << "Saved to: " << filename << "\n";
        }
    }
    return all_ok;
}

int main()
{
    InferenceRunner r;
    r.init_model("./models/lama_fp32.onnx");
    cv::Mat img = cv::imread("./images/input_image.jpg");
    cv::Mat msk = cv::imread("./images/dilated_mask.png", cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        std::cerr << "image not found: " << "\n";
        return 2;
    }
    if (msk.empty())
    {
        std::cerr << "mask not found: " << "\n";
        return 2;
    }

    auto outs = r.run(img, msk);
    save_pngs(outs, "./outputs", "output");
    return 0;
}
