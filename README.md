
# Car Detection from Video

This project uses background subtraction to detect moving cars in videos. It processes video files from a specified dataset folder, detects vehicles, and writes the output to a new video file. The background subtractor model is saved after processing for future use.

## Requirements

- Python 3.x
- OpenCV (`cv2` package)
- NumPy

To install the required dependencies, use the following command:

```bash
pip install opencv-python numpy
```

## File Structure

```
.
├── car_detector_config.json      # Configuration file for dataset and output folders
├── car_detection.py             # Python script for car detection
├── datasets/                    # Folder containing input video files (.mp4)
├── processed_videos/            # Folder for saving the processed output videos
└── mog2_state.npy               # Saved background subtractor model state (if available)
```

## Configuration

The configuration is stored in a JSON file (`car_detector_config.json`), which includes:

- **dataset_folder**: Path to the folder containing the input video files.
- **output_folder**: Path to the folder where processed video files will be saved.
- **min_contour_area**: Minimum area for detected contours to be considered cars.
- **model_state_file**: Path to save/load the background subtractor model state.

Example of `car_detector_config.json`:

```json
{
    "dataset_folder": "C:\Object Model Classification\datasets",
    "output_folder": "C:\Object Model Classification\processed_videos",
    "min_contour_area": 500,
    "model_state_file": "mog2_state.npy"
}
```

## How It Works

1. **Background Subtraction**: The script uses OpenCV's `BackgroundSubtractorMOG2` to perform background subtraction on video frames to identify moving objects.
2. **Contour Detection**: It then detects contours in the foreground mask, and if the contour area is above the specified threshold (`min_contour_area`), it is considered a detected car.
3. **Bounding Box**: A green bounding box is drawn around each detected car, and the label "Car" is displayed.
4. **Output**: The processed video with detected cars is saved in the specified output folder.

## Running the Script

1. Place your input `.mp4` videos in the `datasets` folder.
2. Ensure that the configuration file (`car_detector_config.json`) is properly set up.
3. Run the script `car_detection.py`:

```bash
python main.py
```

The script will process all `.mp4` files in the `datasets` folder, detect cars, and save the results in the `processed_videos` folder. It will also save the background subtractor model for future use.

## Notes

- The script assumes that the videos are in `.mp4` format.
- You can adjust the `min_contour_area` parameter to fine-tune the car detection based on video quality.
- The background subtractor model (`mog2_state.npy`) is saved after processing for efficient reuse.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
