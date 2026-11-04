/* tiny_nn.c - see tiny_nn.h. ANSI C99, no libc, no malloc, no globals. */
#include "tiny_nn.h"

#define TN_NULL ((void *)0)

/* ------------------------------------------------------------- arena ----- */
static uint32_t tn_align_up(uint32_t n)
{
    return (n + (TN_ARENA_ALIGN - 1u)) & ~(TN_ARENA_ALIGN - 1u);
}

void tn_arena_init(tn_arena *a, void *buf, uint32_t size)
{
    a->base = (uint8_t *)buf;
    a->size = size;
    a->used[0] = a->used[1] = 0;
    a->peak = 0;
}

void tn_arena_reset(tn_arena *a)
{
    a->used[0] = a->used[1] = 0;
}

void *tn_arena_alloc(tn_arena *a, int end, uint32_t bytes)
{
    uint32_t n = tn_align_up(bytes);
    uint32_t total = a->used[0] + a->used[1];
    void *p;
    if (n > a->size || total > a->size - n) {
        return TN_NULL;
    }
    a->used[end] += n;
    total += n;
    if (total > a->peak) {
        a->peak = total;
    }
    p = (end == 0) ? (void *)(a->base + a->used[0] - n)
                   : (void *)(a->base + a->size - a->used[1]);
    return p;
}

void tn_arena_release(tn_arena *a, int end)
{
    a->used[end] = 0; /* each end holds at most one live tensor */
}

/* ----------------------------------------------------------- helpers ----- */
/* Round-half-up fixed-point rescale: (acc * mult + 2^(rs-1)) >> rs. */
static __inline int32_t tn_requant(int32_t acc, int32_t mult, uint32_t rs, int32_t zp)
{
    int64_t v = (int64_t)acc * (int64_t)mult;
    v += (int64_t)1 << (rs - 1u);
    v >>= rs;
    v += zp;
    if (v > 127) {
        v = 127;
    } else if (v < -128) {
        v = -128;
    }
    return (int32_t)v;
}

/* Signed 4-bit weight at flat element index i (low nibble first). */
static __inline int32_t tn_w4(const uint8_t *w, uint32_t i)
{
    uint32_t b = w[i >> 1];
    uint32_t nib = (i & 1u) ? (b >> 4) : (b & 0xFu);
    return (int32_t)(nib ^ 8u) - 8;
}

/* acc = bias + sum_j (x[j] - in_zp) * w[row*len + j] */
static int32_t tn_dot(const tn_linear *p, uint32_t row, uint32_t len, const int8_t *x)
{
    int32_t acc = p->bias[row];
    uint32_t j;
    int32_t zp = p->in_zp;
    if (p->w_bits == 8u) {
        const int8_t *w = (const int8_t *)p->weights + (uint32_t)row * len;
        for (j = 0; j < len; ++j) {
            acc += ((int32_t)x[j] - zp) * (int32_t)w[j];
        }
    } else {
        const uint8_t *w = (const uint8_t *)p->weights;
        uint32_t base = row * len;
        for (j = 0; j < len; ++j) {
            acc += ((int32_t)x[j] - zp) * tn_w4(w, base + j);
        }
    }
    return acc;
}

/* ------------------------------------------------------------ layers ----- */
void tn_conv1d(tn_ctx *ctx, const tn_linear *p, const int8_t *in, int8_t *out)
{
    uint32_t out_len = tn_conv_out_len(p);
    uint32_t wl = (uint32_t)p->kernel * p->in_ch;
    uint32_t pos, oc;
    for (pos = 0; pos < out_len; ++pos) {
        const int8_t *win = in + (uint32_t)pos * p->stride * p->in_ch;
        for (oc = 0; oc < p->out_ch; ++oc) {
            int32_t acc = tn_dot(p, oc, wl, win);
            out[pos * p->out_ch + oc] =
                (int8_t)tn_requant(acc, p->mult[oc], p->rshift[oc], p->out_zp);
        }
    }
    ctx->macs += out_len * p->out_ch * wl;
}

void tn_dense(tn_ctx *ctx, const tn_linear *p, const int8_t *in, int8_t *out)
{
    uint32_t oc;
    uint32_t n = p->in_ch;
    for (oc = 0; oc < p->out_ch; ++oc) {
        int32_t acc = tn_dot(p, oc, n, in);
        out[oc] = (int8_t)tn_requant(acc, p->mult[oc], p->rshift[oc], p->out_zp);
    }
    ctx->macs += (uint32_t)p->out_ch * n;
}

void tn_relu(int8_t *buf, uint32_t n, int8_t zp)
{
    uint32_t i;
    for (i = 0; i < n; ++i) {
        if (buf[i] < zp) {
            buf[i] = zp;
        }
    }
}

uint32_t tn_argmax(const int8_t *buf, uint32_t n)
{
    uint32_t i, best = 0;
    for (i = 1; i < n; ++i) {
        if (buf[i] > buf[best]) {
            best = i;
        }
    }
    return best;
}

/* ------------------------------------------------------------ driver ----- */
tn_status tn_run(const tn_model *m, tn_ctx *ctx, const int8_t *input,
                 uint32_t *class_out, int8_t *logits_out)
{
    int8_t *cur;
    int cur_end = 0;
    uint32_t cur_len = m->input_len, i;
    int have_class = 0;

    if (m == TN_NULL || ctx == TN_NULL || input == TN_NULL) {
        return TN_ERR_ARG;
    }
    ctx->macs = 0;
    tn_arena_reset(&ctx->arena);

    cur = (int8_t *)tn_arena_alloc(&ctx->arena, 0, m->input_len); /* stage the sample in SRAM */
    if (cur == TN_NULL) {
        return TN_ERR_ARENA;
    }
    for (i = 0; i < m->input_len; ++i) {
        cur[i] = input[i];
    }

    for (i = 0; i < m->n_layers; ++i) {
        const tn_layer *L = &m->layers[i];
        if (L->type == TN_CONV1D || L->type == TN_DENSE) {
            const tn_linear *p = L->lin;
            uint32_t out_len = (L->type == TN_CONV1D) ? tn_conv_out_len(p) * p->out_ch : p->out_ch;
            int nxt = 1 - cur_end;
            int8_t *out = (int8_t *)tn_arena_alloc(&ctx->arena, nxt, out_len);
            if (out == TN_NULL) {
                return TN_ERR_ARENA;
            }
            if (L->type == TN_CONV1D) {
                tn_conv1d(ctx, p, cur, out);
            } else {
                tn_dense(ctx, p, cur, out);
            }
            tn_arena_release(&ctx->arena, cur_end);
            cur = out;
            cur_end = nxt;
            cur_len = out_len;
        } else if (L->type == TN_RELU) {
            tn_relu(cur, cur_len, L->zp);
        } else if (L->type == TN_ARGMAX) {
            if (class_out != TN_NULL) {
                *class_out = tn_argmax(cur, cur_len);
            }
            have_class = 1;
        } else {
            return TN_ERR_ARG;
        }
    }
    if (logits_out != TN_NULL) {
        for (i = 0; i < m->num_classes && i < cur_len; ++i) {
            logits_out[i] = cur[i];
        }
    }
    return have_class ? TN_OK : TN_ERR_ARG;
}
