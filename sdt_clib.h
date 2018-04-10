#ifndef _SDT_CLIB_H_
#define _SDT_CLIB_H_

#include <stdio.h>
#include "sdt_infer.h"

#ifdef __cplusplus
extern "C" {
#endif

	void sdt_init();
	void sdt_detect(unsigned char *input, int height, int width, int x_shift, int y_shift,
			char *res_str, FILE *res_fp, struct predictions **res_preds);
	void sdt_cleanup();
	float sdt_get_time_detect();
	float sdt_get_time_misc();

#ifdef __cplusplus
}
#endif

#endif	/* _SDT_CLIB_H_ */
