#include "sdt_clib.h"

void sdt_init(const char *wts)
{
	sdt_infer_init(wts);
}

void sdt_detect(unsigned char *input, int height, int width, int x_shift, int y_shift,
		char *res_str, FILE *res_fp, struct predictions **res_preds)
{
	sdt_infer_detect(input, height, width, x_shift, y_shift,
			res_str, res_fp, res_preds);
}

void sdt_cleanup()
{
	sdt_infer_cleanup();
}

float sdt_get_time_detect()
{
	return sdt_infer_get_time_detect();
}

float sdt_get_time_misc()
{
	return sdt_infer_get_time_misc();
}
