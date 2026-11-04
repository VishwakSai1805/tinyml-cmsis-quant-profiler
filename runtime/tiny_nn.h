/*
 * tiny_nn.h - ANSI C99 fixed-point inference engine for microcontrollers.
 *
 *  - integer-only (INT8 activations, INT8 or packed-INT4 weights, INT32 accumulate)
 *  - ZERO heap use: every activation lives in a caller-supplied static arena
 *  - layers: Conv1D, Dense, ReLU, ArgMax
 *
 * Arena model: a double-ended bump allocator. Activation tensors ping-pong
 * between the low and the high end, so a layer needs only (input + output)
 * bytes. `peak` records the exact high-water mark in bytes.
 */
#ifndef TINY_NN_H
#define TINY_NN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TN_ARENA_ALIGN 4u

typedef enum {
    TN_OK = 0,
    TN_ERR_ARENA = -1, /* arena too small for this model */
    TN_ERR_ARG = -2    /* malformed model / null pointer  */
} tn_status;

typedef struct {
    uint8_t *base;
    uint32_t size;
    uint32_t used[2]; /* [0] low end (grows up), [1] high end (grows down) */
    uint32_t peak;    /* max(used[0] + used[1]) ever observed */
} tn_arena;

void tn_arena_init(tn_arena *a, void *buf, uint32_t size);
/* Free everything but keep the recorded peak. */
void tn_arena_reset(tn_arena *a);
/* end: 0 = low, 1 = high. Returns NULL if it does not fit. */
void *tn_arena_alloc(tn_arena *a, int end, uint32_t bytes);
void tn_arena_release(tn_arena *a, int end);

/* Quantised Conv1D / Dense parameters (channel-last, valid padding). */
typedef struct {
    uint16_t in_len;  /* positions (Dense: 1)                               */
    uint16_t in_ch;   /* channels  (Dense: number of input features)        */
    uint16_t out_ch;
    uint16_t kernel;  /* Dense: 1                                           */
    uint16_t stride;
    uint8_t  w_bits;  /* 8 or 4 (4 = two weights per byte, low nibble first) */
    int8_t   in_zp;
    int8_t   out_zp;
    const void    *weights; /* int8_t[out_ch][kernel*in_ch] or packed uint8_t */
    const int32_t *bias;    /* int32[out_ch]                                  */
    const int32_t *mult;    /* Q31 multiplier per output channel              */
    const uint8_t *rshift;  /* total right shift per output channel           */
} tn_linear;

typedef enum { TN_CONV1D = 0, TN_DENSE, TN_RELU, TN_ARGMAX } tn_layer_type;

typedef struct {
    tn_layer_type type;
    const tn_linear *lin; /* CONV1D / DENSE */
    int8_t zp;            /* RELU: clamp floor (the zero-point)             */
} tn_layer;

typedef struct {
    const tn_layer *layers;
    uint32_t n_layers;
    uint32_t input_len;   /* int8 elements */
    uint32_t num_classes;
} tn_model;

typedef struct {
    tn_arena arena;
    uint32_t macs; /* MACs executed by the last tn_run (exact) */
} tn_ctx;

static __inline uint32_t tn_conv_out_len(const tn_linear *p)
{
    return (uint32_t)((p->in_len - p->kernel) / p->stride + 1u);
}

/* Layer kernels (exposed for unit testing). `in`/`out` are int8 [len][ch]. */
void tn_conv1d(tn_ctx *ctx, const tn_linear *p, const int8_t *in, int8_t *out);
void tn_dense(tn_ctx *ctx, const tn_linear *p, const int8_t *in, int8_t *out);
void tn_relu(int8_t *buf, uint32_t n, int8_t zp);
/* First maximum wins (same as numpy.argmax). */
uint32_t tn_argmax(const int8_t *buf, uint32_t n);

/* Full forward pass. input: m->input_len int8. logits_out (optional, caller
 * buffer of m->num_classes) receives the final int8 logits. */
tn_status tn_run(const tn_model *m, tn_ctx *ctx, const int8_t *input,
                 uint32_t *class_out, int8_t *logits_out);

#ifdef __cplusplus
}
#endif
#endif /* TINY_NN_H */
