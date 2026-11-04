#include <stdio.h>
#include <stdlib.h>
#include "plat.h"
void plat_puts(const char *s) { fputs(s, stdout); }
void plat_exit(int code) { fflush(stdout); exit(code); }
