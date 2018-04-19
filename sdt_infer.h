#ifndef _SDT_INFER_H_
#define _SDT_INFER_H_

#include <stdio.h>

struct predictions {
	float *klass;
	float *prob;
	float *bbox;
	int *keep;
	int num;
};

#ifdef __cplusplus
extern "C" {
#endif

void sdt_infer_init(const char *wts);
void sdt_infer_detect(unsigned char *input, int height, int width, int x_shift, int y_shift,
		char *res_str, FILE *res_fp, struct predictions **res_preds);
void sdt_infer_cleanup(void);
float sdt_infer_get_time_detect(void);
float sdt_infer_get_time_misc(void);

#ifdef __cplusplus
}
#endif

#endif	/* _SDT_INFER_H_ */
