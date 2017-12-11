#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <err.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "trtUtil.h"

static const char *imageFormat[] = {"jpeg", "jpg", "png", "ppm", "bmp", NULL};

static int isImageFile(char *name)
{
     assert(strlen(name) > 0);
     char *suffix;
     int i;

     if ((suffix = strrchr(name, '.')) == NULL)
          return 0;
     for (i = 0; imageFormat[i]; i++)
          if (strstr(suffix, imageFormat[i]))
               return 1;
     return 0;
}

static char *changeSuffix(char *name, const char *new_suffix)
{
     assert(strlen(new_suffix) > 0 && strlen(name) > 0);
     char *suffix;

     if ((suffix = strrchr(name, '.')) == NULL) {
          suffix = name + strlen(name);
          *suffix = '.';
     }
     suffix++;
     strcpy(suffix, new_suffix);

     return name;
}

std::vector<std::string> getImageList(const char *pathname)
{
     struct dirent *dirp;
     DIR *dp;
     std::vector<std::string> imgList;

     // if (stat(pathname, &statbuf) < 0) {
     //      perror("stat failed");
     //      exit(EXIT_FAILURE);
     // }
     // if (S_ISDIR(statbuf.st_mode) == 0) {
     //      fprintf(stderr, "'%s' not a directory.\n", pathname);
     //      exit(EXIT_FAILURE);
     // }
     if ((dp = opendir(pathname)) == NULL)
          err(EXIT_FAILURE, "%s", pathname);

     while ((dirp = readdir(dp)) != NULL) {
          if (!isImageFile(dirp->d_name) || !strcmp(dirp->d_name, ".") || !strcmp(dirp->d_name, ".."))
               continue;
          imgList.push_back(std::string(pathname) + std::string("/") + std::string(dirp->d_name));
     }
     closedir(dp);
     return imgList;
}

char *sprintResultFilePath(char *buf, const char *img_name, const char *res_dir)
{
     DIR *dp;
     if ((dp = opendir(res_dir)) == NULL)
          err(EXIT_FAILURE, "%s", res_dir);
     closedir(dp);

     char *img_name_cpy = (char *)malloc(sizeof(char) * (strlen(img_name) + 1));
     strcpy(img_name_cpy, img_name);
     char *file_name;
     if ((file_name = strrchr(img_name_cpy, '/')) == NULL)
          file_name = img_name_cpy;
     else
          file_name++;
     sprintf(buf, "%s/%s", res_dir, file_name);
     changeSuffix(buf, "txt");
     free(img_name_cpy);
     return buf;
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

cv::Mat readImage(const std::string& filename, int width, int height, float *img_width, float *img_height)
{
    cv::Mat img = cv::imread(filename);
    printf("filename: %s\t", filename.c_str());
    printf("img.total(): %ld\t", img.total());
    if (img_width && img_height) {
         *img_width = img.size().width;
         *img_height = img.size().height;
    }
    cv::resize(img, img, cv::Size(width, height));
    return img;
}
