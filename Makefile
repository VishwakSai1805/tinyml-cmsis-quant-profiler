# Build the benchmark for the host (gcc) or Cortex-M3 (arm-none-eabi-gcc, QEMU).
#   make host  MODEL_DIR=build/m8 [ARENA_SIZE=4096]
#   make qemu  MODEL_DIR=build/m8 [ARENA_SIZE=4096]   (build + run on QEMU)
MODEL_DIR  ?= build/model
OUT        ?= build/out
ARENA_SIZE ?= 4096

WARN   = -std=c99 -pedantic -Wall -Wextra -Werror -Wconversion -Wno-sign-conversion
INC    = -Iruntime -Ibench -I$(MODEL_DIR)
DEFS   = -DTN_ARENA_SIZE=$(ARENA_SIZE)u
CORE   = runtime/tiny_nn.c

ARM_CC = arm-none-eabi-gcc
ARM_FLAGS = -mcpu=cortex-m3 -mthumb -Os -ffreestanding -fno-builtin \
            -fno-tree-loop-distribute-patterns -ffunction-sections -fdata-sections \
            -DTN_USE_SECTIONS

.PHONY: host qemu runtime-obj clean
host: $(OUT)/bench_host
	$(OUT)/bench_host

$(OUT)/bench_host: bench/bench.c bench/plat_host.c $(CORE) | $(OUT)
	gcc $(WARN) -O2 $(INC) $(DEFS) $^ -o $@

qemu: $(OUT)/bench_m3.elf
	timeout 60 qemu-system-arm -M lm3s6965evb -nographic -semihosting-config enable=on,target=native -kernel $<

$(OUT)/bench_m3.elf: bench/bench.c bench/plat_qemu.c bench/startup_m3.c $(CORE) bench/link_m3.ld | $(OUT)
	$(ARM_CC) $(ARM_FLAGS) $(WARN) $(INC) $(DEFS) -nostdlib -T bench/link_m3.ld \
	    -Wl,--gc-sections -Wl,-Map=$(OUT)/bench_m3.map \
	    bench/bench.c bench/plat_qemu.c bench/startup_m3.c $(CORE) -lgcc -o $@

# Runtime-only object (for an exact code-size measurement of the engine itself).
runtime-obj: $(OUT)/tiny_nn_m3.o
$(OUT)/tiny_nn_m3.o: $(CORE) | $(OUT)
	$(ARM_CC) $(ARM_FLAGS) $(WARN) -Iruntime -c $< -o $@

$(OUT):
	mkdir -p $(OUT)

clean:
	rm -rf build
