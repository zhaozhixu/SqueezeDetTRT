#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <dirent.h>
#include <unistd.h>
#include <err.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>

#include "sdt_clib.h"
#include "trtUtil.h"
#include "sdt_alloc.h"

static const int OUTPUT_BBOX_SIZE = 4;
static const float PLOT_PROB_THRESH = 0.3;

static const char *CLASS_NAMES[] = {"car", "person", "riding", "bike_riding", "boat", "truck", "horse_riding"};

#define IMG_NAME_SIZE_GUESS 1024

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

static std::vector<std::string> getImageList(const char *pathname, const char *eval_list)
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

static void drawBbox(cv::Mat &frame, struct predictions *preds)
{
     assert(!frame.empty() && preds->bbox && preds->klass && preds->prob && preds->keep);
     int i;
     char *prob_s = (char *)sdt_alloc(32);
     float *bbox;
     for (i = 0; i < preds->num; i++) {
          if (!preds->keep[i] || preds->prob[i] < PLOT_PROB_THRESH)
               continue;
          bbox = &preds->bbox[i * OUTPUT_BBOX_SIZE];
          cv::rectangle(frame, cv::Point(bbox[0], bbox[1]), cv::Point(bbox[2], bbox[3]), cv::Scalar(0, 255, 0));
          sprintf(prob_s, "%.2f", preds->prob[i]);
          cv::putText(frame, std::string(CLASS_NAMES[(int)preds->klass[i]]) + ": " + std::string(prob_s), cv::Point(bbox[0], bbox[1]), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
     }
     sdt_free(prob_s);
}


static void fprintResult(FILE *fp, struct predictions *preds)
{
     assert(fp && preds->bbox && preds->klass && preds->prob && preds->keep);

     int i;
     float *bbox;
     for (i = 0; i < preds->num; i++) {
          if (!preds->keep[i])
               continue;
          bbox = &preds->bbox[i * OUTPUT_BBOX_SIZE];
          fprintf(fp, "%s -1 -1 0.0 %.2f %.2f %.2f %.2f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %.3f\n",
                  CLASS_NAMES[(int)preds->klass[i]], bbox[0], bbox[1], bbox[2], bbox[3], preds->prob[i]);
     }
}

static const struct option longopts[] = {
     {"eval-list", 1, NULL, 'e'},
     {"video", 1, NULL, 'v'},
     {"ground-truth", 1, NULL, 'g'},
     {"bbox-dir", 1, NULL, 'b'},
     {"x-shift", 1, NULL, 'x'},
     {"y-shift", 1, NULL, 'y'},
     {"help", 0, NULL, 'h'},
     {0, 0, 0, 0}
};

static const char *usage = "Usage: sqdtrt [options] IMAGE_DIR RESULT_DIR\n\
Apply SqueezeDet detection algorithm to images in IMAGE_DIR.\n\
Print detection results to one text file per image in RESULT_DIR using KITTI dataset format.\n\
\n\
Options:\n\
       -e, --eval-list=EVAL_LIST_FILE          Provide an evaluation list file which contains\n\
                                               the image names (without extension names)\n\
                                               in IMAGE_DIR to be evaluated.\n\
       -v, --video=VIDEO_FILE                  Detect a video file and play detected video\n\
                                               in a new window. IMAGE_DIR and RESULT_DIR\n\
                                               are not needed.\n\
       -g, --ground-truth=GROUND_TRUTH_DIR     Draw ground truth in red in the video.\n\
       -b, --bbox-dir=BBOX_DIR                 Draw bounding boxes in images or video and\n\
                                               save them in BBOX_DIR.\n\
       -x, --x-shift=X_SHIFT                   Shift all bboxes downward X_SHIFT pixels.\n\
       -y, --y-shift=Y_SHIFT                   Shift all bboxes rightward Y_SHIFT pixels.\n\
       -h, --help                              Print this help and exit.\n";

static void print_usage_and_exit()
{
     fputs(usage, stderr);
     exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
     int opt, optindex;
     char *img_dir = NULL, *result_dir = NULL, *eval_list = NULL,
          *gt_dir = NULL, *video = NULL, *bbox_dir = NULL;
     int x_shift = 0, y_shift = 0;
     while ((opt = getopt_long(argc, argv, ":e:v:b:x:y:h", longopts, &optindex)) != -1) {
          switch (opt) {
          case 'e':
               eval_list = optarg;
               break;
          case 'v':
               video = optarg;
               break;
          case 'g':
               gt_dir = optarg;
               break;
          case 'b':
               bbox_dir = optarg;
               break;
          case 'x':
               x_shift = atoi(optarg);
               break;
          case 'y':
               y_shift = atoi(optarg);
               break;
          case 'h':
               print_usage_and_exit();
               break;
          case ':':
               fprintf(stderr, "option --%s needs a value\n", longopts[optindex].name);
               break;
          case '?':
               fprintf(stderr, "unknown option %c\n", optopt);
               break;
          }
     }
     if (video == NULL && optind >= argc)
          print_usage_and_exit();
     if (video == NULL) {
          img_dir = argv[optind++];
          result_dir = argv[optind];
          validateDir(img_dir, 0);
          validateDir(result_dir, 1);
     }
     if (bbox_dir != NULL)
          validateDir(bbox_dir, 1);

     // initialize library
     sdt_init();

     // read image or video, alloc path buffer
     FILE *result_fp;
     char *result_file_path = NULL;
     char *img_name_buf = NULL;
     char *bbox_file_path = NULL;
     std::vector<std::string> imageList;
     int img_list_size;
     // double write_fps;
     cv::VideoCapture cap;
     cv::VideoWriter writer;
     cv::Mat frame;
     if (video == NULL) {
          result_file_path = sdt_path_alloc(NULL);
          img_name_buf = sdt_path_alloc(NULL);
          imageList = getImageList(img_dir, eval_list);
          img_list_size = imageList.size();
          printf("number of images: %d\n", img_list_size);
     } else {
          cap = cv::VideoCapture(video);
          if (!cap.isOpened()) {
               fprintf(stderr, "error reading video file: %s\n", video);
               exit(EXIT_FAILURE);
          }
          if (bbox_dir != NULL) {
               bbox_file_path = sdt_path_alloc(NULL);
               assemblePath(bbox_file_path, bbox_dir, video, "_bbox.avi");
               if (!writer.open(bbox_file_path,
                                CV_FOURCC('M', 'J', 'P', 'G'),
                                cap.get(CV_CAP_PROP_FPS),
                                cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH),
                                         cap.get(CV_CAP_PROP_FRAME_HEIGHT)), true)) {
                    fprintf(stderr, "error open cv::VideoWriter for file: %s\n", bbox_file_path);
                    exit(EXIT_FAILURE);
               }
               sdt_free(bbox_file_path);
          }
     }

     // do inference
     int frame_idx = 0;
     char key;
     double start_fps, end_fps, start_imread, end_imread;
     double fps, timeImread, timeDetect, timeMisc;
     double imread_time_sum = 0, detect_time_sum = 0, misc_time_sum = 0, fps_sum = 0;
     struct predictions *preds;
     for (;; frame_idx++) {
          start_fps = getUnixTime();
          start_imread = start_fps;
          if (video == NULL) {
               if (frame_idx >= img_list_size) { // end of images
                    frame_idx--;
                    break;
               }
               getFileName(img_name_buf, imageList[frame_idx].c_str());
               printf("(%d/%d) image: %s ", frame_idx+1, img_list_size, img_name_buf);
               frame = cv::imread(imageList[frame_idx]);
               if (frame.empty()) {
                    fprintf(stderr, "error reading image\n");
                    continue;
               }
          } else {
               if (cap.read(frame) == false) { // end of video
                    frame_idx--;
                    break;
               }
               if (frame.empty()) {
                    fprintf(stderr, "error reading frame %d\n", frame_idx);
                    continue;
               }
          }
          end_imread = getUnixTime();

          sdt_detect(frame.data, frame.size().height, frame.size().width,
                     x_shift, y_shift, NULL, NULL, &preds);

          if (video == NULL) {
               assemblePath(result_file_path, result_dir, imageList[frame_idx].c_str(), ".txt");
               result_fp = fopen(result_file_path, "w");
               fprintResult(result_fp, preds);
               fclose(result_fp);
          } else {
               drawBbox(frame, preds);
               // fprintResult(stdout, preds);
               if (bbox_dir != NULL) {
                    writer.write(frame);
               }
               cv::imshow("detection", frame);
               key = cv::waitKey(1);
               if (key == ' ') {
                    cv::waitKey(0);
               } else if (key == 'q' || key == 27) { // 27 is the ASCII code of ESC
                    frame_idx--;
                    break;
               }
          }

          end_fps = getUnixTime();
          fps = 1 / (end_fps - start_fps);
          timeImread = (end_imread - start_imread) * 1000;
          timeDetect = sdt_get_time_detect();
          timeMisc = sdt_get_time_misc();
          printf("imread: %.2fms detect: %.2fms misc: %.2fms fps: %.2fHz\n", timeImread, timeDetect, timeMisc, fps);
          imread_time_sum += timeImread;
          detect_time_sum += timeDetect;
          misc_time_sum += timeMisc;
          fps_sum += fps;
     }
     cap.release();
     writer.release();

     // compute timing result
     double avg_imread, avg_detect, avg_misc, avg_fps;
     avg_imread = imread_time_sum / (frame_idx + 1);
     avg_detect = detect_time_sum / (frame_idx + 1);
     avg_misc = misc_time_sum / (frame_idx + 1);
     avg_fps = fps_sum / (frame_idx + 1);
     printf("Average timing: imread: %.2fms detect: %.2fms misc: %.2fms fps: %.2fHz\n", avg_imread, avg_detect, avg_misc, avg_fps);

     // cleanup library
     sdt_cleanup();

     return 0;
}
