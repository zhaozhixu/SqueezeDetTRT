#ifndef _SDT_ALLOC_H_
#define _SDT_ALLOC_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

	void *sdt_alloc(size_t size);
	char *sdt_path_alloc(size_t *sizep);

#ifdef __cplusplus
}
#endif

#define sdt_free free

#endif  /* _SDT_ALLOC_H_ */
