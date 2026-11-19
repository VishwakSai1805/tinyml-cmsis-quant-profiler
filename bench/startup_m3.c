/* Minimal Cortex-M3 start-up: vector table, .data copy, .bss clear. */
#include <stdint.h>

extern uint32_t _estack, _sidata, _sdata, _edata, _sbss, _ebss;
extern int main(void);

void Reset_Handler(void);
void Default_Handler(void) { for (;;) { } }

__attribute__((section(".isr_vector"), used))
const uintptr_t g_vectors[] = {
    (uintptr_t)&_estack, (uintptr_t)Reset_Handler,
    (uintptr_t)Default_Handler, (uintptr_t)Default_Handler, (uintptr_t)Default_Handler,
    (uintptr_t)Default_Handler, (uintptr_t)Default_Handler,
};

void Reset_Handler(void)
{
    uint32_t *s = &_sidata, *d = &_sdata;
    while (d < &_edata) { *d++ = *s++; }
    for (d = &_sbss; d < &_ebss; ) { *d++ = 0; }
    (void)main();
    for (;;) { }
}
