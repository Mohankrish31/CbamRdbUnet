🚧 CBAM-RDB-UNet (WIP)
Work-in-progress: Low-light colonoscopy image enhancement using CBAM and Residual Dense Blocks (RDB) within a U-Net framework.

✅ Current Goals
✅ Build model: CBAM_RDB_UNet

✅ Train on CVC-ColonDB dataset

✅ Integrate multi-loss:

Structural Similarity (SSIM)

Learned Perceptual Image Patch Similarity (LPIPS)

Sobel Edge Loss

Mean Squared Error (MSE)

📊 Evaluation Metrics
Metric	Status
Total Loss	✅ Done
C-PSNR	❌ Pending
SSIM	❌ Pending
EBCM	❌ Pending
LPIPS	❌ Pending

🧠 Architecture Summary
The CBAM-RDB-UNet architecture combines:

U-Net for semantic segmentation-style image translation.

Residual Dense Blocks (RDB) for better feature representation and gradient flow.

Convolutional Block Attention Module (CBAM) for spatial and channel-wise attention enhancement.

This hybrid model is designed to handle extremely low-light colonoscopy images, boosting both perceptual quality and structural fidelity.

🔄 Next Steps
 Finalize validation loop and compute all metrics.

 Save trained .pt model.

 Run inference on real unseen test images.

 Plot training curves:

Total Loss

SSIM

LPIPS

EBCM

C-PSNR

 Complete full documentation with results and visual comparisons.

 Finalize README.md and publish baseline results.

