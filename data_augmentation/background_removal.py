import cv2
import mediapipe as mp
import numpy as np
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def main(file_name, output_file):
    cap = cv2.VideoCapture(file_name)
    if not cap.isOpened():
        print(f'Failed to open video: {file_name}')
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height), isColor=True)  # noqa: E501

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:  # noqa: E501
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if we reach the end of the video

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = selfie_segmentation.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Create an alpha channel based on the segmentation mask
            alpha_channel = (results.segmentation_mask > 0.1) * 255
            # Stack the channels together to form an RGBA image
            rgba_image = np.concatenate((image, alpha_channel[..., None]), axis=-1).astype(np.uint8)
            # Optionally convert the image back to BGR (with an alpha channel) for saving with OpenCV
            bgra_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)

            # # Create a 4-channel image (including alpha channel for transparency)
            # alpha_channel = np.ones(results.segmentation_mask.shape) * (results.segmentation_mask > 0.1)
            # rgba_image = cv2.merge((image, alpha_channel.astype(np.uint8) * 255))

            # Write the frame to the output video
            out.write(bgra_image)

    cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video using MediaPipe Selfie Segmentation.')  # noqa: E501
    parser.add_argument('file_name', help='The name of the video file to process.')
    parser.add_argument('output_file', help='The name of the output video file.')
    args = parser.parse_args()
    main(args.file_name, args.output_file)