#ifndef _SDT_CLIB_H_
#define _SDT_CLIB_H_

#include <stdio.h>
#include "sdt_infer.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the CUDA engine, TensorRT engine and neural network etc.
 * @param    wts is the path of the weight file.
 */
void sdt_init(const char *wts);

/**
 * Detection function.
 * @param    input is the image data, which should be a continous byte array of
 *           height * width * 3 elements. Note that the	array dimision should
 *           be in [H W C] order.
 * @param    height is the height of the image.
 * @param    width is the width of the image.
 * @param    x_shift is the horizonal shift of bounding boxes set manually.
 * @param    y_shift is the vertical shift of bounding boxes set manually.
 * @param    res_str is the output string whose format is "class -1 -1 0.0 xmin
 *           ymin xmax ymax 0.0 0.0 0.0 0.0 0.0 0.0 0.0 prob" compatible with
 *           KITTI dataset format.
 * @param    res_fp is the output stream pointer, to which the output string
 *           will be printed to.
 * @param    *res_preds is the output structure pointer which will save the
 *           output.
 */
void sdt_detect(unsigned char *input, int height, int width, int x_shift, int y_shift,
		char *res_str, FILE *res_fp, struct predictions **res_preds);

/**
 * Cleanup function which should be called when the detection is done.
 */
void sdt_cleanup(void);

/**
 * Function used to get the detection time in milliseconds.
 */
float sdt_get_time_detect(void);

/**
 * Function used to get the miscellaneous time in milliseconds.
 */
float sdt_get_time_misc(void);

#ifdef __cplusplus
}
#endif

#endif	/* _SDT_CLIB_H_ */
