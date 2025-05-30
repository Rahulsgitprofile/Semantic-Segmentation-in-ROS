# Semantic Image Segmentation with U-Net (TensorFlow)



Semantic segmentation is a fundamental task in computer vision where each pixel of an image is assigned to a class label. This project builds a semantic segmentation pipeline using the U-Net architecture in TensorFlow, trained on the KITTI Segmentation Dataset.


📂 Dataset
  Dataset used: KITTI Segmentation Dataset
  Input: RGB images
  Labels: Color-coded masks (each color represents a semantic class)


🧱 Model Architecture
We use a configurable U-Net architecture:
  Encoder: Downsampling path with Conv2D, BatchNorm, Dropout, and MaxPooling
  Decoder: Upsampling path using Conv2DTranspose and skip connections
  Output: 30-class pixel-wise classification using softmax


Training:
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=[SparseMeanIoU(num_classes=30)]
)
model.fit(train, validation_data=val, epochs=1)

