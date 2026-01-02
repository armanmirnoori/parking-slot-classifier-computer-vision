## Workflow & How It Works (Simplified)

The model training process is based entirely on labeled images from the `clf-data/` directory.

**Training Process**
   - Images are passed through the network in 'img.py' script
   - Loss is calculated (difference between prediction and real label)
   - Backpropagation updates model weights
   - The final trained model is saved as `model.p`

**Parking Slot Selection**
   The script `area_selector.py` lets the user mark parking slots manually, storing coordinates in `slots.json`.

**Prediction / Inference**
   `predict.py`:
   - Reads slot regions from `slots.json`
   - Extracts each region from input image/video
   - Passes them into the trained model
   - Outputs classification results visually

## Optimal Input Format
This model works best when the parking area is filmed using a **bird’s-eye (top-down overhead) view**, where the entire parking zone is visible from above and slots are clearly separated.


## Sample Result
![Parking Slot Classification Result](sample-result/overview.png
)




## Credits
- Training code and implementation inspiration: **@ComputerVisionEngineer** (YouTube)
- Dataset used for training: labeled images inside `clf-data/`, originally provided by the same channel.
- Evaluation and test video: bird’s-eye (top-down overhead) parking footage from the creator.

## Reference Video
The original video used in this project belongs to the creator and is publicly available here:

YOUTUBER_VIDEO_LINK_HERE: https://drive.google.com/drive/folders/1jovc7oBMFV1DutrijBFDbEMIO7fWwh5o?usp=drive_link

## Clarification of Ownership
This repository includes my own explanation and integration for better understanding.  
**Only the model training process and training-related code are based on the original creator’s work.**  
No other sections of the project claim ownership of his content.
