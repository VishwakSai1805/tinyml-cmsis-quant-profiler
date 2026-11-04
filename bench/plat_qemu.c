/* ARM semihosting platform layer (QEMU `-semihosting`). */
#include <stdint.h>
#include "plat.h"

static int sh_call(int op, const void *arg)
{
    register int r0 __asm__("r0") = op;
    register const void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

void plat_puts(const char *s) { (void)sh_call(0x04 /* SYS_WRITE0 */, s); }

void plat_exit(int code)
{
    /* AArch32 SYS_EXIT_EXTENDED: {reason, exit status} */
    static volatile uint32_t blk[2];
    blk[0] = 0x20026u; /* ADP_Stopped_ApplicationExit */
    blk[1] = (uint32_t)code;
    (void)sh_call(0x20 /* SYS_EXIT_EXTENDED */, (const void *)blk);
    for (;;) { }
}
