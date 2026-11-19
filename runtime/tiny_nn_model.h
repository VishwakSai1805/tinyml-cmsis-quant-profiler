/* tiny_nn_model.h - glue: builds the static layer graph from a generated
 * model_weights.h. Include in exactly one translation unit. */
#ifndef TINY_NN_MODEL_H
#define TINY_NN_MODEL_H
#include "model_weights.h"
#include "tiny_nn.h"

#define TN_LINEAR(P, name)                                                        \
    { TN_##P##_IN_LEN, TN_##P##_IN_CH, TN_##P##_OUT_CH, TN_##P##_K, TN_##P##_STRIDE, \
      TN_##P##_WBITS, TN_##P##_IN_ZP, TN_##P##_OUT_ZP, name##_w, name##_b,        \
      name##_mult, name##_rshift }

/* The fc layer is exported as 1 position x (flattened) features. */
static const tn_linear tn_lin_conv1 = TN_LINEAR(CONV1, conv1);
static const tn_linear tn_lin_conv2 = TN_LINEAR(CONV2, conv2);
static const tn_linear tn_lin_fc    = TN_LINEAR(FC, fc);

static const tn_layer tn_layers[] = {
    { TN_CONV1D, &tn_lin_conv1, 0 },
    { TN_RELU,   0, TN_CONV1_OUT_ZP },
    { TN_CONV1D, &tn_lin_conv2, 0 },
    { TN_RELU,   0, TN_CONV2_OUT_ZP },
    { TN_DENSE,  &tn_lin_fc, 0 },
    { TN_ARGMAX, 0, 0 },
};

static const tn_model tn_model_desc = {
    tn_layers, (uint32_t)(sizeof tn_layers / sizeof tn_layers[0]),
    TN_INPUT_LEN * TN_INPUT_CH, TN_NUM_CLASSES
};
#endif
