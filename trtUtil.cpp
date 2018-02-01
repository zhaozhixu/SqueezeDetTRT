#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <err.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "trtUtil.h"
#include "sdt_alloc.h"
#include "NvInfer.h"

#define IMG_NAME_SIZE_GUESS 1024

static const char *imageFormat[] = {"jpeg", "jpg", "png", "ppm", "bmp", NULL};

using namespace nvinfer1;

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

char *changeSuffix(char *name, const char *new_suffix)
{
     assert(name && new_suffix);
     char *suffix;

     if ((suffix = strrchr(name, '.')) == NULL) {
          suffix = name + strlen(name);
     }
     strcpy(suffix, new_suffix);

     return name;
}

char *getFileName(char *buf, const char *path)
{
     assert(buf);
     char *name, *path_copy;

     path_copy = (char *)sdt_alloc(strlen(path) + 1);
     strcpy(path_copy, path);
     if ((name = strrchr(path_copy, '/')) == NULL)
          name = path_copy;
     else
          name++;
     strcpy(buf, name);
     sdt_free(path_copy);

     return buf;
}

char *joinPath(char *prepath, char *subpath)
{
     assert(prepath && subpath);
     char *pre_tail;

     pre_tail = prepath + strlen(prepath) - 1;
     if (*pre_tail != '/')
          *++pre_tail = '/';
     strcpy(++pre_tail, subpath);

     return prepath;
}

char *assemblePath(char *buf, const char *dir, const char *file_path, const char *suffix)
{
     char *file_path_cpy = (char *)sdt_alloc(strlen(file_path) + 1);
     strcpy(file_path_cpy, file_path);
     char *file_name;
     if ((file_name = strrchr(file_path_cpy, '/')) == NULL)
          file_name = file_path_cpy;
     else
          file_name++;
     strcpy(buf, dir);
     joinPath(buf, file_name);
     changeSuffix(buf, suffix);
     sdt_free(file_path_cpy);
     return buf;
}

void validateDir(const char *dir, int do_mkdir)
{
     DIR *dp;
     if ((dp = opendir(dir)) == NULL) {
          if (do_mkdir) {
               if (mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH) == -1)
                    err(EXIT_FAILURE, "%s", dir);
          } else {
               err(EXIT_FAILURE, "%s", dir);
          }
     } else {
          closedir(dp);
     }
}

double getUnixTime(void)
{
     struct timespec tv;

     if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;

     return (tv.tv_sec + (tv.tv_nsec / 1.0e9));
}

std::vector<std::string> getImageList(const char *pathname, const char *eval_list)
{
     long img_name_size;
     char *img_name;
     struct dirent *dirp;
     DIR *dp;
     FILE *fp;
     std::vector<std::string> imgList;
     std::map<std::string, int> evalMap;

     if ((img_name_size = pathconf(pathname, _PC_NAME_MAX)) == -1)
          img_name_size = IMG_NAME_SIZE_GUESS;
     img_name = (char *)sdt_alloc(img_name_size);

     if (eval_list) {
          if ((fp = fopen(eval_list, "r")) == NULL)
               err(EXIT_FAILURE, "%s", eval_list);
          while (fscanf(fp, "%s", img_name) != EOF) {
               evalMap[std::string(img_name)] = 1;
          }
          fclose(fp);
     }
     if ((dp = opendir(pathname)) == NULL)
          err(EXIT_FAILURE, "%s", pathname);

     while ((dirp = readdir(dp)) != NULL) {
          if (!isImageFile(dirp->d_name) || !strcmp(dirp->d_name, ".") || !strcmp(dirp->d_name, ".."))
               continue;
          if (eval_list) {
               strcpy(img_name, dirp->d_name);
               changeSuffix(img_name, "");
               if (evalMap.count(std::string(img_name)) > 0)
                    imgList.push_back(std::string(pathname) + std::string("/") + std::string(dirp->d_name));
               continue;
          }
          imgList.push_back(std::string(pathname) + std::string("/") + std::string(dirp->d_name));
     }
     sdt_free(img_name);
     closedir(dp);
     return imgList;
}

// // not used
// cv::Mat readImage(const std::string& filename, int width, int height, float *img_width, float *img_height)
// {
//      cv::Mat img = cv::imread(filename);
//      if (img.empty())
//           return img;

//      if (img_width && img_height) {
//           *img_width = img.size().width;
//           *img_height = img.size().height;
//      }
//      cv::resize(img, img, cv::Size(width, height));
//      return img;
// }

// void preprocessFrame(cv::Mat &frame, cv::Mat &frame_origin, int width, int height, float *img_width, float *img_height)
// {
//      assert(!frame_origin.empty());

//      if (img_width && img_height) {
//           *img_width = frame_origin.size().width;
//           *img_height = frame_origin.size().height;
//      }
//      cv::resize(frame_origin, frame, cv::Size(width, height));
// }

// // Our weight files are in a very simple space delimited format.
// // [type] [size] <data x size in hex>
// std::map<std::string, Weights> loadWeights(const std::string file)
// {
//     std::map<std::string, Weights> weightMap;
// 	std::ifstream input(file);
// 	assert(input.is_open() && "Unable to load weight file.");
//     int32_t count;
//     input >> count;
//     assert(count > 0 && "Invalid weight map file.");
//     while(count--) {
//         Weights wt{DataType::kFLOAT, nullptr, 0};
//         uint32_t type, size;
//         std::string name;
//         input >> name >> std::dec >> type >> size;
//         wt.type = static_cast<DataType>(type);
//         if (wt.type == DataType::kFLOAT) {
//             uint32_t *val = reinterpret_cast<uint32_t*>(sdt_alloc(sizeof(val) * size)); // TODO: wrong sizeof oprand
//             for (uint32_t x = 0, y = size; x < y; ++x)
//             {
//                 input >> std::hex >> val[x];

//             }
//             wt.values = val;
//         } else if (wt.type == DataType::kHALF) {
//             uint16_t *val = reinterpret_cast<uint16_t*>(sdt_alloc(sizeof(val) * size)); // wrong sizeof oprand
//             for (uint32_t x = 0, y = size; x < y; ++x)
//             {
//                 input >> std::hex >> val[x];
//             }
//             wt.values = val;
//         }
//         wt.count = size;
//         weightMap[name] = wt;
//     }
//     return weightMap;
// }
