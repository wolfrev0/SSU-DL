# dl.rs
Deep learning implementation from scratch by Rust

This project was developed for term project of Deep Learning Course in Soongsil University.

## Implemented operations
- identity
- matmul
- relu
- eltw_add
- eltw_mult
- softmax_xy
- softmax_y
- softmax_cross_entropy
- transpose
- layer_norm
- concat4
- attention
- 4_head_attention
- encoder

## Getting Started
1. Install rust and cargo.
2. Clone this repository to your local machine.
3. run `cargo run --bin mnist`

## src/bin/mnist.rs Output
```
##### Test #####
Cross Entropy Error: 2.0229712
##### Learn #####
----- epoch 0 -----
Cross Entropy Error (validation): 0.84213644
----- epoch 1 -----
Cross Entropy Error (validation): 0.29854906
----- epoch 2 -----
Cross Entropy Error (validation): 0.028893478
----- epoch 3 -----
Cross Entropy Error (validation): 0.06684602
----- epoch 4 -----
Cross Entropy Error (validation): 0.01479665
##### Test #####
Cross Entropy Error (test): 0.0011330106
```

## Report
https://docs.google.com/document/d/1CtcD79BKG_9Bd9S9RZ5lz2RGdapsw-nDpxiD_O2_K0Y/edit?usp=sharing