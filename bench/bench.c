/*
 * bench.c - on-target benchmark harness (host and Cortex-M3/QEMU).
 *
 * Runs every held-out test vector through tiny_nn, and prints KEY=VALUE lines:
 *   ARENA_SIZE / ARENA_PEAK  static SRAM arena and exact high-water mark (bytes)
 *   MACS                     MAC operations per inference (counted at run time)
 *   ACC_INT / ACC_FP32       correct predictions out of SAMPLES (int engine vs
 *                            FP32 reference predictions exported with the data)
 *   REF_MISMATCH             disagreements with the bit-exact Python reference
 *   LOGIT_CHECKSUM           rolling checksum of all int8 logits
 */
#include "tiny_nn_model.h"
#include "test_vectors.h"
#include "plat.h"

#ifndef TN_ARENA_SIZE
#define TN_ARENA_SIZE 4096u
#endif

static uint8_t g_arena[TN_ARENA_SIZE] __attribute__((aligned(8)));

static void put_kv(const char *k, uint32_t v)
{
    char buf[12];
    int n = 0, i;
    plat_puts(k);
    plat_puts("=");
    if (v == 0) {
        buf[n++] = '0';
    }
    while (v) {
        buf[n++] = (char)('0' + v % 10u);
        v /= 10u;
    }
    {
        char out[12];
        for (i = 0; i < n; ++i) {
            out[i] = buf[n - 1 - i];
        }
        out[n] = 0;
        plat_puts(out);
    }
    plat_puts("\n");
}

int main(void)
{
    tn_ctx ctx;
    uint32_t i, k;
    uint32_t acc_int = 0, acc_fp = 0, mismatch = 0, cs = 0, macs = 0, macs_bad = 0;
    int8_t logits[TN_NUM_CLASSES];

    tn_arena_init(&ctx.arena, g_arena, TN_ARENA_SIZE);
    plat_puts("MODEL=" TN_MODEL_NAME "\n");
    put_kv("ARENA_SIZE", TN_ARENA_SIZE);
    put_kv("SAMPLES", TN_TEST_N);

    for (i = 0; i < TN_TEST_N; ++i) {
        uint32_t cls = 0;
        tn_status st = tn_run(&tn_model_desc, &ctx, &tn_test_x[i * TN_INPUT_LEN], &cls, logits);
        if (st != TN_OK) {
            put_kv("ERROR", (uint32_t)(-st));
            put_kv("ARENA_PEAK", ctx.arena.peak);
            plat_exit(2);
        }
        if (i == 0) {
            macs = ctx.macs;
        } else if (ctx.macs != macs) {
            ++macs_bad;
        }
        acc_int += (cls == tn_test_y[i]);
        acc_fp += (tn_test_fp32_pred[i] == tn_test_y[i]);
        mismatch += (cls != tn_test_ref_pred[i]);
        for (k = 0; k < TN_NUM_CLASSES; ++k) {
            cs = cs * 31u + (uint32_t)(uint8_t)logits[k];
        }
    }
    put_kv("ARENA_PEAK", ctx.arena.peak);
    put_kv("MACS", macs);
    put_kv("MACS_VARIANT", macs_bad);
    put_kv("ACC_INT", acc_int);
    put_kv("ACC_FP32", acc_fp);
    put_kv("REF_MISMATCH", mismatch);
    put_kv("LOGIT_CHECKSUM", cs);
    put_kv("CHECKSUM_OK", cs == TN_REF_LOGIT_CHECKSUM);
    plat_puts("DONE\n");
    plat_exit(mismatch == 0 && cs == TN_REF_LOGIT_CHECKSUM ? 0 : 1);
    return 0;
}
