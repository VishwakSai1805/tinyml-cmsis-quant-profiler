/* TI LM3S6965 (QEMU lm3s6965evb): 256 KiB flash @0x0, 64 KiB SRAM @0x20000000 */
ENTRY(Reset_Handler)
MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 256K
  SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 64K
}
_estack = ORIGIN(SRAM) + LENGTH(SRAM);
SECTIONS
{
  .text : { KEEP(*(.isr_vector)) *(.text*) *(.rodata*) . = ALIGN(4); } > FLASH
  /* Model payload (weights, bias, requant params) kept apart so its Flash
   * footprint can be read straight from the ELF section table. */
  .model_rodata : { *(.model_rodata*) . = ALIGN(4); } > FLASH
  .testdata     : { *(.testdata*) . = ALIGN(4); } > FLASH
  _sidata = LOADADDR(.data);
  .data : { _sdata = .; *(.data*) . = ALIGN(4); _edata = .; } > SRAM AT > FLASH
  .bss  : { _sbss = .; *(.bss*) *(COMMON) . = ALIGN(4); _ebss = .; } > SRAM
  /DISCARD/ : { *(.ARM.exidx*) *(.comment) }
}
