#ifndef _SDT_ALLOC_H_
#define _SDT_ALLOC_H_

#include <stdlib.h>

void *sdt_alloc(size_t size);
char *sdt_path_alloc(size_t *sizep);

#define sdt_free free

#endif  /* _SDT_ALLOC_H_ */
