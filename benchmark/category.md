# 📊 **WORKLOAD SHAPE CATALOG** (Concise Format)

**Generated**: 2025-08-18
**Format**: Concise shape specifications following WORKLOAD_CATALOG.md template
**Coverage**: All workloads organized by operator type with essential shape information

**Total Workloads**: 386 across 28 operator types

---

## ADD Operations

- case1: [32, 512, 768] + [32, 512, 768] --> [32, 512, 768] (BERT)
- case2: [128, 128, 28, 28] + [128, 128, 28, 28] --> [128, 128, 28, 28] (CNN)
- case3: [64, 1280, 7, 7] + [64, 1280, 7, 7] --> [64, 1280, 7, 7] (EfficientNet)
- case4: [16, 1024, 1536] + [16, 1024, 1536] --> [16, 1024, 1536] (GPT)
- case5: [64, 100, 256] + [64, 100, 256] --> [64, 100, 256] (LSTM)
- case6: [128, 96, 112, 112] + [128, 96, 112, 112] --> [128, 96, 112, 112] (MobileNet)
- case7: [64, 256, 56, 56] + [64, 256, 56, 56] --> [64, 256, 56, 56] (ResNet)
- case8: [32, 512, 2048] + [32, 512, 2048] --> [32, 512, 2048] (Transformer)
- case9: [16, 512, 32, 32] + [16, 512, 32, 32] --> [16, 512, 32, 32] (U-Net)
- case10: [32, 197, 768] + [32, 197, 768] --> [32, 197, 768] (ViT)
- case11: [dim1, dim2, dim3, dim4] + [dim1, dim2, dim3, dim4] --> [dim1, dim2, dim3, dim4] (Dynamic)
- case12: [batch, 256, 256] + [batch, 256, 256] --> [batch, 256, 256] (Dynamic)
- case13: [batch, seq_len, hidden] + [1, 1, hidden] --> [batch, seq_len, hidden] (Dynamic)
- case14: [16, channels, 224, 224] + [16, channels, 224, 224] --> [16, channels, 224, 224] (Dynamic)
- case15: [32, 128, hidden_dim] + [32, 128, hidden_dim] --> [32, 128, hidden_dim] (Dynamic)
- case16: [8, 64, height, width] + [8, 64, height, width] --> [8, 64, height, width] (Dynamic)
- case17: [8, dim1, dim2, 16] + [8, dim1, dim2, 16] --> [8, dim1, dim2, 16] (Dynamic)
- case18: [32, seq_len, 768] + [32, seq_len, 768] --> [32, seq_len, 768] (Dynamic)
- case19: [batch, seq_len, hidden_dim] + [batch, seq_len, hidden_dim] --> [batch, seq_len, hidden_dim] (Dynamic)
- case20: [32, seq_len, 768] + [32, seq_len, 768] --> [32, seq_len, 768] (Dynamic)

## MATMUL Operations

- case1: [4, 3] + [3, 5] --> [4, 5] (Static)
- case2: [32, 512, 768] + [768, 768] --> [32, 512, 768] (BERT)
- case3: [32, 512, 30522] + [30522, 768] --> [32, 512, 768] (BERT)
- case4: [256, 512] + [512, 10] --> [256, 10] (CNN)
- case5: [64, 1280] + [1280, 1000] --> [64, 1000] (EfficientNet)
- case6: [16, 1024, 1536] + [1536, 6144] --> [16, 1024, 6144] (GPT)
- case7: [64, 100, 300] + [300, 256] --> [64, 100, 256] (LSTM)
- case8: [32, 512, 512] + [512, 512] --> [32, 512, 512] (General)
- case9: [128, 2048] + [2048, 1000] --> [128, 1000] (ResNet)
- case10: [64, 512, 512] + [512, 2048] --> [64, 512, 2048] (Transformer)
- case11: [32, num_heads, 512, 64] + [32, num_heads, 64, 512] --> [32, num_heads, 512, 512] (Attention)
- case12: [batch_size, 2048] + [2048, 1000] --> [batch_size, 1000] (Dynamic)
- case13: [32, 197, embed_dim] + [embed_dim, 3072] --> [32, 197, 3072] (Dynamic)
- case14: [16, 1024, d_model] + [d_model, 4096] --> [16, 1024, 4096] (Dynamic)
- case15: [64, 100, 300] + [300, hidden_size] --> [64, 100, hidden_size] (Dynamic)
- case16: [16, scale_features, 512] + [512, 256] --> [16, scale_features, 256] (Dynamic)
- case17: [128, 1280] + [1280, num_classes] --> [128, num_classes] (Dynamic)
- case18: [32, seq_len, 768] + [768, 768] --> [32, seq_len, 768] (Dynamic)
- case19: [64, time_steps, 256] + [256, 128] --> [64, time_steps, 128] (Dynamic)
- case20: [32, 512, vocab_size] + [vocab_size, 768] --> [32, 512, 768] (Dynamic)

## CONV2D Operations

- case1: [1, 3, 32, 32] --> [1, 16, 32, 32] (Static)
- case2: [1, 3, 32, 32] --> [1, 16, 32, 32] (Static)
- case3: [1, 3, 32, 32] --> [1, 16, 32, 32] (Static)
- case4: [64, 128, 32, 32] --> [64, 32, 32, 32] (Static)
- case5: [64, 40, 56, 56] --> [64, 240, 56, 56] (EfficientNet)
- case6: [32, 192, 28, 28] --> [32, 64, 28, 28] (Static)
- case7: [128, 32, 112, 112] --> [128, 32, 112, 112] (MobileNet)
- case8: [32, 256, 56, 56] --> [32, 64, 56, 56] (ResNet)
- case9: [64, 3, 224, 224] --> [64, 64, 112, 112] (ResNet)
- case10: [8, 256, 64, 64] --> [8, 21, 64, 64] (Static)
- case11: [32, 512, attn_h, attn_w] --> [32, 1024, attn_h, attn_w] (Attention)
- case12: [batch_size, 64, 56, 56] --> [batch_size, 128, 56, 56] (Dynamic)
- case13: [16, 256, pyramid_h, pyramid_w] --> [16, 256, pyramid_h, pyramid_w] (Dynamic)
- case14: [32, 128, height, width] --> [32, 256, height, width] (Dynamic)
- case15: [128, in_channels, 112, 112] --> [128, 64, 112, 112] (Dynamic)
- case16: [64, 64, 56, 56] --> [64, 128, 56, 56] (Dynamic)
- case17: [8, 256, scale_h, scale_w] --> [8, 512, scale_h, scale_w] (Dynamic)
- case18: [64, 256, 28, 28] --> [64, out_channels, 28, 28] (Dynamic)
- case19: [16, 64, 224, 224] --> [16, 128, output_h, output_w] (Dynamic)
- case20: [32, 128, 112, 112] --> [32, 256, output_h, output_w] (Dynamic)

## RELU Operations

- case1: [32, 512, 768] --> [32, 512, 768] (BERT)
- case2: [128, 128, 28, 28] --> [128, 128, 28, 28] (CNN)
- case3: [64, 1280, 7, 7] --> [64, 1280, 7, 7] (EfficientNet)
- case4: [16, 1024, 4096] --> [16, 1024, 4096] (GPT)
- case5: [64, 100, 256] --> [64, 100, 256] (LSTM)
- case6: [128, 96, 112, 112] --> [128, 96, 112, 112] (MobileNet)
- case7: [64, 256, 56, 56] --> [64, 256, 56, 56] (ResNet)
- case8: [32, 512, 2048] --> [32, 512, 2048] (Transformer)
- case9: [16, 512, 32, 32] --> [16, 512, 32, 32] (U-Net)
- case10: [32, 197, 3072] --> [32, 197, 3072] (ViT)
- case11: [batch, features] --> [batch, features] (Dynamic)
- case12: [32, num_heads, 512, 64] --> [32, num_heads, 512, 64] (Attention)
- case13: [batch_size, 256, 56, 56] --> [batch_size, 256, 56, 56] (Dynamic)
- case14: [batch_size, 1280, height, width] --> [batch_size, 1280, height, width] (Dynamic)
- case15: [128, channels, 112, 112] --> [128, channels, 112, 112] (Dynamic)
- case16: [32, 197, embed_dim] --> [32, 197, embed_dim] (Dynamic)
- case17: [16, 1024, d_model] --> [16, 1024, d_model] (Dynamic)
- case18: [64, 128, height, width] --> [64, 128, height, width] (Dynamic)
- case19: [16, 512, scale_h, scale_w] --> [16, 512, scale_h, scale_w] (Dynamic)
- case20: [32, seq_len, 768] --> [32, seq_len, 768] (Dynamic)

## SIGMOID Operations

- case1: [32, 512, 768] --> [32, 512, 768] (BERT)
- case2: [128, 128, 28, 28] --> [128, 128, 28, 28] (CNN)
- case3: [64, 1280, 7, 7] --> [64, 1280, 7, 7] (EfficientNet)
- case4: [16, 1024, 4096] --> [16, 1024, 4096] (GPT)
- case5: [64, 100, 256] --> [64, 100, 256] (LSTM)
- case6: [128, 96, 112, 112] --> [128, 96, 112, 112] (MobileNet)
- case7: [64, 256, 56, 56] --> [64, 256, 56, 56] (ResNet)
- case8: [32, 512, 2048] --> [32, 512, 2048] (Transformer)
- case9: [16, 512, 32, 32] --> [16, 512, 32, 32] (U-Net)
- case10: [32, 197, 3072] --> [32, 197, 3072] (ViT)
- case11: [32, num_heads, 512, 64] --> [32, num_heads, 512, 64] (Attention)
- case12: [batch_size, 256, 56, 56] --> [batch_size, 256, 56, 56] (Dynamic)
- case13: [batch_size, 1280, height, width] --> [batch_size, 1280, height, width] (Dynamic)
- case14: [128, channels, 112, 112] --> [128, channels, 112, 112] (Dynamic)
- case15: [32, 197, embed_dim] --> [32, 197, embed_dim] (Dynamic)
- case16: [16, 1024, d_model] --> [16, 1024, d_model] (Dynamic)
- case17: [64, 128, height, width] --> [64, 128, height, width] (Dynamic)
- case18: [16, 512, scale_h, scale_w] --> [16, 512, scale_h, scale_w] (Dynamic)
- case19: [32, seq_len, 768] --> [32, seq_len, 768] (Dynamic)
- case20: [64, time_steps, 256] --> [64, time_steps, 256] (Dynamic)

## SOFTMAX Operations

- case1: [32, 512, 768] --> [32, 512, 768] (BERT)
- case2: [128, 128, 28, 28] --> [128, 128, 28, 28] (CNN)
- case3: [64, 1280, 7, 7] --> [64, 1280, 7, 7] (EfficientNet)
- case4: [16, 1024, 4096] --> [16, 1024, 4096] (GPT)
- case5: [64, 100, 256] --> [64, 100, 256] (LSTM)
- case6: [128, 96, 112, 112] --> [128, 96, 112, 112] (MobileNet)
- case7: [64, 256, 56, 56] --> [64, 256, 56, 56] (ResNet)
- case8: [32, 512, 2048] --> [32, 512, 2048] (Transformer)
- case9: [16, 512, 32, 32] --> [16, 512, 32, 32] (U-Net)
- case10: [32, 197, 3072] --> [32, 197, 3072] (ViT)
- case11: [batch, seq_len, vocab_size] --> [batch, seq_len, vocab_size] (Dynamic)
- case12: [32, num_heads, 512, 64] --> [32, num_heads, 512, 64] (Attention)
- case13: [batch_size, 256, 56, 56] --> [batch_size, 256, 56, 56] (Dynamic)
- case14: [batch_size, 1280, height, width] --> [batch_size, 1280, height, width] (Dynamic)
- case15: [128, channels, 112, 112] --> [128, channels, 112, 112] (Dynamic)
- case16: [32, 197, embed_dim] --> [32, 197, embed_dim] (Dynamic)
- case17: [16, 1024, d_model] --> [16, 1024, d_model] (Dynamic)
- case18: [64, 128, height, width] --> [64, 128, height, width] (Dynamic)
- case19: [16, 512, scale_h, scale_w] --> [16, 512, scale_h, scale_w] (Dynamic)
- case20: [32, seq_len, 768] --> [32, seq_len, 768] (Dynamic)

## GELU Operations

- case1: [4, 768] --> [4, 768] (Static)
- case2: [32, 512, 768] --> [32, 512, 768] (BERT)
- case3: [128, 128, 28, 28] --> [128, 128, 28, 28] (CNN)
- case4: [64, 1280, 7, 7] --> [64, 1280, 7, 7] (EfficientNet)
- case5: [16, 1024, 4096] --> [16, 1024, 4096] (GPT)
- case6: [64, 100, 256] --> [64, 100, 256] (LSTM)
- case7: [128, 96, 112, 112] --> [128, 96, 112, 112] (MobileNet)
- case8: [64, 256, 56, 56] --> [64, 256, 56, 56] (ResNet)
- case9: [32, 512, 2048] --> [32, 512, 2048] (Transformer)
- case10: [16, 512, 32, 32] --> [16, 512, 32, 32] (U-Net)
- case11: [32, num_heads, 512, 64] --> [32, num_heads, 512, 64] (Attention)
- case12: [batch_size, 256, 56, 56] --> [batch_size, 256, 56, 56] (Dynamic)
- case13: [batch_size, 1280, height, width] --> [batch_size, 1280, height, width] (Dynamic)
- case14: [128, channels, 112, 112] --> [128, channels, 112, 112] (Dynamic)
- case15: [32, 197, embed_dim] --> [32, 197, embed_dim] (Dynamic)
- case16: [16, 1024, d_model] --> [16, 1024, d_model] (Dynamic)
- case17: [64, 128, height, width] --> [64, 128, height, width] (Dynamic)
- case18: [16, 512, scale_h, scale_w] --> [16, 512, scale_h, scale_w] (Dynamic)
- case19: [32, seq_len, 768] --> [32, seq_len, 768] (Dynamic)
- case20: [64, time_steps, 256] --> [64, time_steps, 256] (Dynamic)

## BATCH_NORM Operations

- case1: [8, 16, 32, 32] --> [8, 16, 32, 32] (Static)
- case2: [8, 16, 32, 32] + [16] + [16] + ... (5 inputs) --> [8, 16, 32, 32] (Static)
- case3: [8, 16, 32, 32] --> [8, 16, 32, 32] (Static)
- case4: [32, 512, 768] --> [32, 512, 768] (BERT)
- case5: [128, 128, 28, 28] --> [128, 128, 28, 28] (CNN)
- case6: [64, 1280, 7, 7] --> [64, 1280, 7, 7] (EfficientNet)
- case7: [16, 1024, 4096] --> [16, 1024, 4096] (GPT)
- case8: [64, 100, 256] --> [64, 100, 256] (LSTM)
- case9: [128, 96, 112, 112] --> [128, 96, 112, 112] (MobileNet)
- case10: [64, 256, 56, 56] --> [64, 256, 56, 56] (ResNet)
- case11: [32, num_heads, 512, 64] --> [32, num_heads, 512, 64] (Attention)
- case12: [batch_size, 256, 56, 56] --> [batch_size, 256, 56, 56] (Dynamic)
- case13: [batch_size, 1280, height, width] --> [batch_size, 1280, height, width] (Dynamic)
- case14: [128, channels, 112, 112] --> [128, channels, 112, 112] (Dynamic)
- case15: [32, 197, embed_dim] --> [32, 197, embed_dim] (Dynamic)
- case16: [16, 1024, d_model] --> [16, 1024, d_model] (Dynamic)
- case17: [64, 128, height, width] --> [64, 128, height, width] (Dynamic)
- case18: [16, 512, scale_h, scale_w] --> [16, 512, scale_h, scale_w] (Dynamic)
- case19: [32, seq_len, 768] --> [32, seq_len, 768] (Dynamic)
- case20: [64, time_steps, 256] --> [64, time_steps, 256] (Dynamic)

## LAYER_NORM Operations

- case1: [32, 512, 768] --> [32, 512, 768] (BERT)
- case2: [128, 128, 28, 28] --> [128, 128, 28, 28] (CNN)
- case3: [64, 1280, 7, 7] --> [64, 1280, 7, 7] (EfficientNet)
- case4: [16, 1024, 4096] --> [16, 1024, 4096] (GPT)
- case5: [64, 100, 256] --> [64, 100, 256] (LSTM)
- case6: [128, 96, 112, 112] --> [128, 96, 112, 112] (MobileNet)
- case7: [64, 256, 56, 56] --> [64, 256, 56, 56] (ResNet)
- case8: [32, 512, 2048] --> [32, 512, 2048] (Transformer)
- case9: [16, 512, 32, 32] --> [16, 512, 32, 32] (U-Net)
- case10: [32, 197, 3072] --> [32, 197, 3072] (ViT)
- case11: [batch, seq_len, hidden_size] + [hidden_size] + [hidden_size] --> [batch, seq_len, hidden_size] (Dynamic)
- case12: [32, num_heads, 512, 64] --> [32, num_heads, 512, 64] (Attention)
- case13: [batch_size, 256, 56, 56] --> [batch_size, 256, 56, 56] (Dynamic)
- case14: [batch_size, 1280, height, width] --> [batch_size, 1280, height, width] (Dynamic)
- case15: [128, channels, 112, 112] --> [128, channels, 112, 112] (Dynamic)
- case16: [32, 197, embed_dim] --> [32, 197, embed_dim] (Dynamic)
- case17: [16, 1024, d_model] --> [16, 1024, d_model] (Dynamic)
- case18: [64, 128, height, width] --> [64, 128, height, width] (Dynamic)
- case19: [16, 512, scale_h, scale_w] --> [16, 512, scale_h, scale_w] (Dynamic)
- case20: [32, seq_len, 768] --> [32, seq_len, 768] (Dynamic)

## RESHAPE Operations

- case1: [32, 512, 768] --> [32, 512, 768] (BERT)
- case2: [128, 128, 28, 28] --> [128, 128, 28, 28] (CNN)
- case3: [64, 1280, 7, 7] --> [64, 1280, 7, 7] (EfficientNet)
- case4: [16, 1024, 4096] --> [16, 1024, 4096] (GPT)
- case5: [64, 100, 256] --> [64, 100, 256] (LSTM)
- case6: [128, 96, 112, 112] --> [128, 96, 112, 112] (MobileNet)
- case7: [64, 256, 56, 56] --> [64, 256, 56, 56] (ResNet)
- case8: [32, 512, 2048] --> [32, 512, 2048] (Transformer)
- case9: [16, 512, 32, 32] --> [16, 512, 32, 32] (U-Net)
- case10: [32, 197, 3072] --> [32, 197, 3072] (ViT)
- case11: [32, num_heads, 512, 64] --> [32, num_heads, 512, 64] (Attention)
- case12: [batch_size, 256, 56, 56] --> [batch_size, 256, 56, 56] (Dynamic)
- case13: [batch_size, 1280, height, width] --> [batch_size, 1280, height, width] (Dynamic)
- case14: [128, channels, 112, 112] --> [128, channels, 112, 112] (Dynamic)
- case15: [32, 197, embed_dim] --> [32, 197, embed_dim] (Dynamic)
- case16: [16, 1024, d_model] --> [16, 1024, d_model] (Dynamic)
- case17: [64, 128, height, width] --> [64, 128, height, width] (Dynamic)
- case18: [16, 512, scale_h, scale_w] --> [16, 512, scale_h, scale_w] (Dynamic)
- case19: [32, seq_len, 768] --> [32, seq_len, 768] (Dynamic)
- case20: [64, time_steps, 256] --> [64, time_steps, 256] (Dynamic)

## TRANSPOSE Operations

- case1: [3, 4, 5] --> [5, 4, 3] (Static)
- case2: [32, 512, 768] --> [32, 512, 768] (BERT)
- case3: [128, 128, 28, 28] --> [128, 128, 28, 28] (CNN)
- case4: [64, 1280, 7, 7] --> [64, 1280, 7, 7] (EfficientNet)
- case5: [16, 1024, 4096] --> [16, 1024, 4096] (GPT)
- case6: [64, 100, 256] --> [64, 100, 256] (LSTM)
- case7: [128, 96, 112, 112] --> [128, 96, 112, 112] (MobileNet)
- case8: [64, 256, 56, 56] --> [64, 256, 56, 56] (ResNet)
- case9: [32, 512, 2048] --> [32, 512, 2048] (Transformer)
- case10: [16, 512, 32, 32] --> [16, 512, 32, 32] (U-Net)
- case11: [32, num_heads, 512, 64] --> [32, num_heads, 512, 64] (Attention)
- case12: [batch_size, 256, 56, 56] --> [batch_size, 256, 56, 56] (Dynamic)
- case13: [batch_size, 1280, height, width] --> [batch_size, 1280, height, width] (Dynamic)
- case14: [128, channels, 112, 112] --> [128, channels, 112, 112] (Dynamic)
- case15: [32, 197, embed_dim] --> [32, 197, embed_dim] (Dynamic)
- case16: [16, 1024, d_model] --> [16, 1024, d_model] (Dynamic)
- case17: [64, 128, height, width] --> [64, 128, height, width] (Dynamic)
- case18: [16, 512, scale_h, scale_w] --> [16, 512, scale_h, scale_w] (Dynamic)
- case19: [32, seq_len, 768] --> [32, seq_len, 768] (Dynamic)
- case20: [64, time_steps, 256] --> [64, time_steps, 256] (Dynamic)

## CONCATENATE Operations

- case1: [32, 512, 768] + [32, 512, 768] --> [32, 512, 1536] (BERT)
- case2: [128, 128, 28, 28] + [128, 256, 28, 28] --> [128, 384, 28, 28] (CNN)
- case3: [64, 1280, 7, 7] + [64, 320, 7, 7] --> [64, 1600, 7, 7] (EfficientNet)
- case4: [16, 512, 1536] + [16, 512, 1536] --> [16, 1024, 1536] (GPT)
- case5: [64, 100, 256] + [64, 100, 256] --> [64, 100, 512] (LSTM)
- case6: [128, 96, 112, 112] + [128, 32, 112, 112] --> [128, 128, 112, 112] (MobileNet)
- case7: [64, 256, 56, 56] + [64, 256, 56, 56] --> [64, 512, 56, 56] (ResNet)
- case8: [32, 512, 64] + [32, 512, 64] + [32, 512, 64] + ... (4 inputs) --> [32, 512, 256] (Transformer)
- case9: [16, 512, 32, 32] + [16, 512, 32, 32] --> [16, 1024, 32, 32] (U-Net)
- case10: [32, 196, 768] + [32, 1, 768] --> [32, 197, 768] (ViT)
- case11: [batch, seq1, hidden] + [batch, seq2, hidden] --> [batch, seq1+seq2, hidden] (Dynamic)
- case12: [32, num_heads, 512, 64] + [32, num_heads, 512, 64] --> [32, num_heads, 512, 128] (Attention)
- case13: [batch_size, 256, 56, 56] + [batch_size, 256, 56, 56] --> [batch_size, 512, 56, 56] (Dynamic)
- case14: [batch_size, 1280, height, width] + [batch_size, 320, height, width] --> [batch_size, 1600, height, width] (Dynamic)
- case15: [128, channels_1, 112, 112] + [128, channels_2, 112, 112] --> [128, total_channels, 112, 112] (Dynamic)
- case16: [32, 197, embed_dim_1] + [32, 197, embed_dim_2] --> [32, 197, total_embed_dim] (Dynamic)
- case17: [16, 1024, d_model_1] + [16, 1024, d_model_2] --> [16, 1024, total_d_model] (Dynamic)
- case18: [64, 128, height, width] + [64, 128, height, width] --> [64, 256, height, width] (Dynamic)
- case19: [16, 512, scale_h, scale_w] + [16, 512, scale_h, scale_w] --> [16, 1024, scale_h, scale_w] (Dynamic)
- case20: [32, seq_len_1, 768] + [32, seq_len_2, 768] --> [32, total_seq_len, 768] (Dynamic)

## SPLIT Operations

- case1: [32, 512, 768] + [32, 512, 768] --> [32, 512, 1536] (BERT)
- case2: [128, 128, 28, 28] + [128, 256, 28, 28] --> [128, 384, 28, 28] (CNN)
- case3: [64, 1280, 7, 7] + [64, 320, 7, 7] --> [64, 1600, 7, 7] (EfficientNet)
- case4: [16, 512, 1536] + [16, 512, 1536] --> [16, 1024, 1536] (GPT)
- case5: [64, 100, 256] + [64, 100, 256] --> [64, 100, 512] (LSTM)
- case6: [128, 96, 112, 112] + [128, 32, 112, 112] --> [128, 128, 112, 112] (MobileNet)
- case7: [64, 256, 56, 56] + [64, 256, 56, 56] --> [64, 512, 56, 56] (ResNet)
- case8: [32, 512, 64] + [32, 512, 64] + [32, 512, 64] + ... (4 inputs) --> [32, 512, 256] (Transformer)
- case9: [16, 512, 32, 32] + [16, 512, 32, 32] --> [16, 1024, 32, 32] (U-Net)
- case10: [32, 196, 768] + [32, 1, 768] --> [32, 197, 768] (ViT)
- case11: [32, num_heads, 512, 64] + [32, num_heads, 512, 64] --> [32, num_heads, 512, 128] (Attention)
- case12: [batch_size, 256, 56, 56] + [batch_size, 256, 56, 56] --> [batch_size, 512, 56, 56] (Dynamic)
- case13: [batch_size, 1280, height, width] + [batch_size, 320, height, width] --> [batch_size, 1600, height, width] (Dynamic)
- case14: [128, channels_1, 112, 112] + [128, channels_2, 112, 112] --> [128, total_channels, 112, 112] (Dynamic)
- case15: [32, 197, embed_dim_1] + [32, 197, embed_dim_2] --> [32, 197, total_embed_dim] (Dynamic)
- case16: [16, 1024, d_model_1] + [16, 1024, d_model_2] --> [16, 1024, total_d_model] (Dynamic)
- case17: [64, 128, height, width] + [64, 128, height, width] --> [64, 256, height, width] (Dynamic)
- case18: [16, 512, scale_h, scale_w] + [16, 512, scale_h, scale_w] --> [16, 1024, scale_h, scale_w] (Dynamic)
- case19: [32, seq_len_1, 768] + [32, seq_len_2, 768] --> [32, total_seq_len, 768] (Dynamic)
- case20: [64, time_steps, 256] + [64, time_steps, 256] --> [64, time_steps, 512] (Dynamic)

## MAX_POOL2D Operations

- case1: [32, 512, 768] --> [32, 512, 768] (BERT)
- case2: [128, 128, 28, 28] --> [128, 128, 28, 28] (CNN)
- case3: [64, 1280, 7, 7] --> [64, 1280, 7, 7] (EfficientNet)
- case4: [16, 1024, 4096] --> [16, 1024, 4096] (GPT)
- case5: [64, 100, 256] --> [64, 100, 256] (LSTM)
- case6: [128, 96, 112, 112] --> [128, 96, 112, 112] (MobileNet)
- case7: [64, 256, 56, 56] --> [64, 256, 56, 56] (ResNet)
- case8: [32, 512, 2048] --> [32, 512, 2048] (Transformer)
- case9: [16, 512, 32, 32] --> [16, 512, 32, 32] (U-Net)
- case10: [32, 197, 3072] --> [32, 197, 3072] (ViT)
- case11: [32, num_heads, 512, 64] --> [32, num_heads, 512, 64] (Attention)
- case12: [batch_size, 256, 56, 56] --> [batch_size, 256, 56, 56] (Dynamic)
- case13: [batch_size, 1280, height, width] --> [batch_size, 1280, height, width] (Dynamic)
- case14: [128, channels, 112, 112] --> [128, channels, 112, 112] (Dynamic)
- case15: [32, 197, embed_dim] --> [32, 197, embed_dim] (Dynamic)
- case16: [16, 1024, d_model] --> [16, 1024, d_model] (Dynamic)
- case17: [64, 128, height, width] --> [64, 128, height, width] (Dynamic)
- case18: [16, 512, scale_h, scale_w] --> [16, 512, scale_h, scale_w] (Dynamic)
- case19: [32, seq_len, 768] --> [32, seq_len, 768] (Dynamic)
- case20: [64, time_steps, 256] --> [64, time_steps, 256] (Dynamic)

## EMBEDDING Operations

- case1: [2, 8] --> [2, 8, 768] (Static)
- case2: [32, 512, 768] --> [32, 512, 768] (BERT)
- case3: [128, 128, 28, 28] --> [128, 128, 28, 28] (CNN)
- case4: [64, 1280, 7, 7] --> [64, 1280, 7, 7] (EfficientNet)
- case5: [16, 1024, 4096] --> [16, 1024, 4096] (GPT)
- case6: [64, 100, 256] --> [64, 100, 256] (LSTM)
- case7: [128, 96, 112, 112] --> [128, 96, 112, 112] (MobileNet)
- case8: [64, 256, 56, 56] --> [64, 256, 56, 56] (ResNet)
- case9: [32, 512, 2048] --> [32, 512, 2048] (Transformer)
- case10: [16, 512, 32, 32] --> [16, 512, 32, 32] (U-Net)
- case11: [32, num_heads, 512, 64] --> [32, num_heads, 512, 64] (Attention)
- case12: [batch_size, 256, 56, 56] --> [batch_size, 256, 56, 56] (Dynamic)
- case13: [batch_size, 1280, height, width] --> [batch_size, 1280, height, width] (Dynamic)
- case14: [128, channels, 112, 112] --> [128, channels, 112, 112] (Dynamic)
- case15: [32, 197, embed_dim] --> [32, 197, embed_dim] (Dynamic)
- case16: [16, 1024, d_model] --> [16, 1024, d_model] (Dynamic)
- case17: [64, 128, height, width] --> [64, 128, height, width] (Dynamic)
- case18: [16, 512, scale_h, scale_w] --> [16, 512, scale_h, scale_w] (Dynamic)
- case19: [32, seq_len, 768] --> [32, seq_len, 768] (Dynamic)
- case20: [64, time_steps, 256] --> [64, time_steps, 256] (Dynamic)

## ATTENTION Subgraph

- case1: [32, 512, 768] x3, [768, 768] x4 -> [32, 512, 768] (BERT Base, 12 heads, static)
- case2: [16, 512, 1024] x3, [1024, 1024] x4 -> [16, 512, 1024] (BERT Large, 16 heads, static)
- case3: [32, 1024, 768] x3, [768, 768] x4 -> [32, 1024, 768] (GPT Small, 12 heads, static)
- case4: [16, 1024, 1024] x3, [1024, 1024] x4 -> [16, 1024, 1024] (GPT Medium, 16 heads, static)
- case5: [32, 197, 768] x3, [768, 768] x4 -> [32, 197, 768] (ViT Base, 12 heads, static)
- case6: [16, 197, 1024] x3, [1024, 1024] x4 -> [16, 197, 1024] (ViT Large, 16 heads, static)
- case7: [16, 512, 768] x3, [768, 768] x4 -> [16, 512, 768] (T5 Base, 12 heads, static)
- case8: [32, 512, 768] x3, [768, 768] x4 -> [32, 512, 768] (RoBERTa Base, 12 heads, static)
- case9: [64, 512, 768] x3, [768, 768] x4 -> [64, 512, 768] (DistilBERT, 12 heads, static)
- case10: [32, 512, 768] x3, [768, 768] x4 -> [32, 512, 768] (ALBERT Base, 12 heads, static)
- case11: [batch_size, 512, 768] x3, [768, 768] x4 -> [batch_size, 512, 768] (Dynamic batch, batch_size: [1,128])
- case12: [32, seq_len, 768] x3, [768, 768] x4 -> [32, seq_len, 768] (Dynamic seq_len, seq_len: [128,2048])
- case13: [16, 512, hidden_dim] x3, [hidden_dim, hidden_dim] x4 -> [16, 512, hidden_dim] (Dynamic hidden_dim, hidden_dim: [768,4096])
- case14: [32, 512, 768] x3, [768, 768] x4 -> [32, 512, 768] (Dynamic num_heads, num_heads: [8,32])
- case15: [32, num_patches, 768] x3, [768, 768] x4 -> [32, num_patches, 768] (Dynamic num_patches, num_patches: [49,784])
- case16: [batch_size, decoder_len, 768], [batch_size, encoder_len, 768] x2, [768, 768] x4 -> [batch_size, decoder_len, 768] (Dynamic encoder-decoder, batch_size: [1,64], encoder_len: [128,1024], decoder_len: [64,512])
- case17: [16, scale_tokens, 512] x3, [512, 512] x4 -> [16, scale_tokens, 512] (Dynamic multi-scale, scale_tokens: [64,1024])
- case18: [batch_size, sparse_len, 1024] x3, [1024, 1024] x4 -> [batch_size, sparse_len, 1024] (Dynamic sparse, batch_size: [1,32], sparse_len: [256,4096])
- case19: [8, long_seq_len, 768] x3, [768, 768] x4 -> [8, long_seq_len, 768] (Dynamic long sequence, long_seq_len: [2048,16384])
- case20: [batch_size, total_tokens, embed_dim] x3, [embed_dim, embed_dim] x4 -> [batch_size, total_tokens, embed_dim] (Dynamic multi-modal, batch_size: [1,32], total_tokens: [256,2048], embed_dim: [768,1536])

## BROADCAST_ADD Operations

- case1: [batch, height, width, 3] + [1, 1, 1, 3] + [batch, 1, 1, 1] --> varies (Dynamic)

## REDUCE_MEAN Operations

- case1: [batch, seq_len, hidden_dim] + [batch, seq_len] --> varies (Dynamic)

## BERT_MODEL Operations

- case1: [batch, seq_len] + [batch, seq_len] + [batch, seq_len] --> [batch, seq_len, 768] (BERT)
- case2: [batch, seq_len] + [batch, seq_len] + [batch, seq_len] --> [batch, seq_len, 1024] (BERT)

## EFFICIENTNET_MODEL Operations

- case1: [batch, 3, 240, 240] --> [batch, 1000] (EfficientNet)
- case2: [batch, 3, 224, 224] --> [batch, 1000] (EfficientNet)

## GPT_MODEL Operations

- case1: [batch, seq_len] + [batch, seq_len] --> [batch, seq_len, 50257] (GPT)
- case2: [batch, seq_len] + [batch, seq_len] --> [batch, seq_len, 50257] (GPT)

## GRU_MODEL Operations

- case1: [batch, seq_len, 128] --> [batch, seq_len, 256] (General)

## LSTM_MODEL Operations

- case1: [batch, seq_len, 128] --> [batch, seq_len, 256] (LSTM)

## MOBILENET_MODEL Operations

- case1: [batch, 3, 224, 224] --> [batch, 1000] (MobileNet)
- case2: [batch, 3, 224, 224] --> [batch, 1000] (MobileNet)

## RESNET_MODEL Operations

- case1: [batch, 3, 224, 224] --> [batch, 1000] (ResNet)
- case2: [batch, 3, 224, 224] --> [batch, 1000] (ResNet)

## TRANSFORMER_ATTENTION Operations

- case1: [2, 8, 64] + [2, 8, 64] + [2, 8, 64] + ... (7 inputs) --> [2, 8, 64] (Attention)
- case2: [2, 8, 64] + [2, 8, 64] + [2, 8, 64] + ... (7 inputs) --> [2, 8, 64] (Attention)
- case3: [2, 8, 64] + [2, 8, 64] + [2, 8, 64] + ... (7 inputs) --> [2, 8, 64] (Attention)

## TRANSFORMER_BLOCK Operations

- case1: [batch, seq_len, 768] + [batch, seq_len, seq_len] + [768, 3072] --> varies (Dynamic)

## UNET_MODEL Operations

- case1: [batch, 3, 256, 256] --> [batch, 1, 256, 256] (U-Net)

## VIT_MODEL Operations

- case1: [batch, 3, 224, 224] --> [batch, 1000] (ViT)
- case2: [batch, 3, 224, 224] --> [batch, 1000] (ViT)

--

## Summary

**Total Coverage**: 386 workloads across 28 operator types
- **Core Operators**: ADD, MATMUL, CONV2D, ReLU, Sigmoid, Softmax, GELU (15 types)
- **Specialized Operations**: Batch Norm, Layer Norm, Reshape, Transpose, Concat, Split, Max Pool, Embedding (13 types)
- **Model-Level Workloads**: Complete architectures for BERT, GPT, ResNet, EfficientNet, ViT, MobileNet, U-Net, LSTM/GRU

**Shape Categories**:
- **Static Shapes**: Fixed tensor dimensions for deterministic inference
- **Dynamic Shapes**: Symbolic dimensions for variable batch sizes, sequence lengths, and feature dimensions
- **Model Shapes**: End-to-end input/output specifications for complete architectures

**Format**: `input_shapes --> output_shape (use_case)`
**Data Types**: float32 (default), int64 (token IDs), specified where different
**Dynamic Shapes**: Symbolic dimensions shown as variable names (e.g., batch_size, seq_len)
**File Organization**: Consolidated in `{operator}_all.py` files with 20 test cases each

---

## Summary

**Total Coverage**: 386 workloads across 28 operator types
- **Core Operators**: ADD, MATMUL, CONV2D, ReLU, Sigmoid, Softmax, GELU (15 types)
- **Specialized Operations**: Batch Norm, Layer Norm, Reshape, Transpose, Concat, Split, Max Pool, Embedding (13 types)
- **Model-Level Workloads**: Complete architectures for BERT, GPT, ResNet, EfficientNet, ViT, MobileNet, U-Net, LSTM/GRU

**Shape Categories**:
- **Static Shapes**: Fixed tensor dimensions for deterministic inference
- **Dynamic Shapes**: Symbolic dimensions for variable batch sizes, sequence lengths, and feature dimensions
- **Model Shapes**: End-to-end input/output specifications for complete architectures

**Format**: `input_shapes --> output_shape (use_case)`
**Data Types**: float32 (default), int64 (token IDs), specified where different
**Dynamic Shapes**: Symbolic dimensions shown as variable names (e.g., batch_size, seq_len)
**File Organization**: Consolidated in `{operator}_all.py` files with 20 test cases each
