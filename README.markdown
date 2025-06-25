# CNN for Patch Image Classification (CNNclassification2.m)

## Description
This MATLAB script trains a convolutional neural network (CNN) to classify patch images stored as `.mat` files, likely derived from remote sensing or marine science datasets. The dataset is organized in subfolders, where each subfolder corresponds to a class label. The script handles data loading, defines a CNN architecture, trains the model, evaluates performance using accuracy, precision, recall, and F1-score, and visualizes results with a confusion matrix and network architecture.

The script is designed for researchers working with satellite or marine ecosystem data, leveraging MATLAB’s Deep Learning Toolbox for model training and evaluation.

## Requirements
- **MATLAB** (R2018b or later recommended)
- **Deep Learning Toolbox**
- **Image Processing Toolbox**
- A dataset of patch images in `.mat` format, organized in subfolders by class labels
- Sufficient storage for saving the trained network (`net.mat`)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/youchulJ/CNN_2022_Matlab.git
   ```
2. Ensure MATLAB is installed with the required toolboxes.
3. Place your dataset in the directory specified in the script (e.g., `H:\2022 대학원\reagain\과학기지\test2`).

## Usage
1. **Update the dataset path**:
   Modify the `imageDatastore` path to point to your dataset:
   ```matlab
   imds = imageDatastore(fullfile('H:\2022 대학원\reagain\과학기지\test2'),...
       'FileExtensions','.mat','ReadFcn',@matReader,"IncludeSubfolders",true,"LabelSource","foldernames");
   ```
   Replace the path with the location of your `.mat` files.

2. **Define the `matReader` function**:
   The script requires a custom `matReader` function to load `.mat` files. Example:
   ```matlab
   function data = matReader(filename)
       loaded = load(filename);
       data = loaded.data; % Adjust 'data' to the variable name in your .mat files
   end
   ```
   Save this function as `matReader.m` in your MATLAB path.

3. **Run the script**:
   Open `CNNclassification2.m` in MATLAB and execute it. The script will:
   - Load and split the dataset (70% training, 30% validation)
   - Define a CNN with three convolutional layers, batch normalization, ReLU activation, and max-pooling
   - Train the model using the Adam optimizer for 10 epochs
   - Save the trained network to `H:\2022 대학원\reagain\과학기지\test2_net\net.mat`
   - Evaluate performance on the validation set, calculating accuracy, precision, recall, and F1-score
   - Visualize results with a confusion matrix and network architecture

4. **Output**:
   - Trained network saved as `net.mat` in the specified `outputDir`
   - Confusion matrix plot (`plotconfusion`)
   - Network architecture plot (`plot(lgraph)`)
   - Command window output: accuracy, precision, recall, and F1-score

## Code Overview
- **Data Loading**:
  - Uses `imageDatastore` to load `.mat` files with a custom `matReader` function.
  - Splits data into 70% training (`imdsTrain`) and 30% validation (`imdsValidation`) sets using `splitEachLabel`.
  - Checks input data dimensions (128×128×10) using `readimage` and `size`.

- **CNN Architecture**:
  - **Input Layer**: Accepts 128×128×10 patch images
  - **Three Convolutional Blocks**:
    - Conv2D (3×3, 8 filters) → Batch Normalization → ReLU → Max Pooling (3×3, stride 3)
    - Conv2D (3×3, 16 filters) → Batch Normalization → ReLU → Max Pooling (3×3, stride 3)
    - Conv2D (3×3, 32 filters) → Batch Normalization → ReLU
  - **Output Layers**: Fully connected layer (3 classes) → Softmax → Classification
  - Total layers defined in the `layers` array.

- **Training**:
  - Optimizer: Adam
  - Initial learning rate: 0.01
  - Epochs: 10
  - Validation frequency: Every 30 iterations
  - Shuffles data every epoch
  - Plots training progress (`Plots','training-progress'`)
  - Uses `auto` execution environment (CPU/GPU based on availability)

- **Evaluation**:
  - Computes validation accuracy using `classify` and `sum(YPred == YValidation)/numel(YValidation)`.
  - Generates a confusion matrix with `confusionmat` and visualizes it using `plotconfusion`.
  - Calculates:
    - **Precision**: Ratio of true positives to predicted positives per class
    - **Recall**: Ratio of true positives to actual positives per class
    - **F1-Score**: Harmonic mean of precision and recall
  - Averages precision and recall across classes for overall metrics.

- **Visualization**:
  - Displays the confusion matrix to show classification performance.
  - Plots the network architecture using `layerGraph` and `plot(lgraph)`.

## Example Dataset Structure
```plaintext
H:\2022 대학원\reagain\과학기지\test2\
├── Class1\
│   ├── image1.mat
│   ├── image2.mat
│   └── ...
├── Class2\
│   ├── image1.mat
│   ├── image2.mat
│   └── ...
└── Class3\
    ├── image1.mat
    ├── image2.mat
    └── ...
```
Each `.mat` file should contain a 128×128×10 array representing a patch image.

## Notes
- **Dataset Compatibility**: Ensure `.mat` files contain data in a 128×128×10 format, matching the input layer. Adjust the `matReader` function to extract the correct variable from your `.mat` files.
- **Number of Classes**: The fully connected layer is set to 3 classes (`fullyConnectedLayer(3)`). Modify this if your dataset has a different number of classes.
- **Output Directory**: Update the `outputDir` path for saving `net.mat`:
  ```matlab
  outputDir = 'H:\2022 대학원\reagain\과학기지\test2_net';
  ```
- **Commented Code**: The script includes commented sections for visualizing random images. Uncomment these to inspect sample images:
  ```matlab
  % figure
  % numImage = img;
  % perm = randperm(numImage,20);
  % for i = 1:20
  %     subplot(4,5,i);
  %     imshow(imds.Files{perm(i)});
  %     drawnow;
  % end
  ```
  Note that `imshow` may not work directly with `.mat` files unless they contain displayable image data.
- **Performance Tuning**: Adjust hyperparameters (e.g., `MaxEpochs`, `InitialLearnRate`, or layer configurations) based on your dataset and computational resources.

## Applications
This script is suitable for remote sensing and marine science applications, such as classifying oceanographic patch images (e.g., sea surface temperature, salinity, or ecosystem patterns) derived from satellite data. The CNN can be adapted for other multi-channel image classification tasks.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for bug reports, feature requests, or improvements.

## License
MIT License