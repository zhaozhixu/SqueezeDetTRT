#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "trtUtil.h"

extern const int INPUT_H;
extern const int INPUT_W;

std::vector<std::string> getImageList(const char *pathname)
{
     struct stat statbuf;
     struct dirent *dirp;
     DIR *dp;
     std::vector<std::string> imgList;

     if (stat(pathname, &statbuf) < 0) {
          perror("stat failed");
          exit(EXIT_FAILURE);
     }
     if (S_ISDIR(statbuf.st_mode) == 0) {
          fprintf(stderr, "'%s' not a directory.\n", pathname);
          exit(EXIT_FAILURE);
     }
     if ((dp = opendir(pathname)) == NULL) {
          perror("cannot read directory");
          exit(EXIT_FAILURE);
     }

     while ((dirp = readdir(dp)) != NULL) {
          if (!strcmp(dirp->d_name, ".") || !strcmp(dirp->d_name, ".."))
               continue;
          // TODO: filter jpeg
          imgList.push_back(std::string(pathname) + std::string("/") + std::string(dirp->d_name));
     }
     return imgList;
}

// Our weight files are in a very simple space delimited format.
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::map<std::string, Weights> weightMap;
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while(count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t type, size;
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);
        if (wt.type == DataType::kFLOAT) {
            uint32_t *val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size)); // wrong sizeof oprand
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];

            }
            wt.values = val;
        } else if (wt.type == DataType::kHALF) {
            uint16_t *val = reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size)); // wrong sizeof oprand
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

cv::Mat readImage(const std::string& filename, int width, int height, float *x_scale, float *y_scale)
{
    cv::Mat img = cv::imread(filename);
    printf("filename: %s\n", filename.c_str());
    printf("img.total(): %ld\n", img.total());
    int img_width = img.size().width;
    int img_height = img.size().height;
    cv::resize(img, img, cv::Size(width, height));
    if (x_scale && y_scale) {
         *x_scale = 1.0 * width / img_width;
         *y_scale = 1.0 * height / img_height;
    }
    return img;
}
