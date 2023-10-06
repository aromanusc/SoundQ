import cv2
import numpy as np

class VideoPixelTransformer:
	def __init__(self, video_path):
		self.cap = cv2.VideoCapture(video_path)
		self.frame_width = int(self.cap.get(3))
		self.frame_height = int(self.cap.get(4))

	def apply_transformation(self, transformation_id):
		if transformation_id == 1:
			return lambda frame: self.transform_1(frame)
		elif transformation_id == 2:
			return lambda frame: self.transform_2(frame)
		elif transformation_id == 3:
			return lambda frame: self.transform_3(frame)
		elif transformation_id == 4:
			return lambda frame: self.transform_4(frame)
		elif transformation_id == 5:
			return lambda frame: self.transform_5(frame)
		elif transformation_id == 6:
			return lambda frame: self.transform_6(frame)
		elif transformation_id == 7:
			return lambda frame: self.transform_7(frame)
		elif transformation_id == 8:
			return lambda frame: self.transform_8(frame)
		else:
			raise ValueError("Invalid transformation ID")

	def transform_1(self, frame):
		frame = np.roll(frame, shift=1440, axis=1)
		frame = np.flip(frame, axis=0)
		return frame

	def transform_2(self, frame):
		frame = np.flip(frame, axis=1)
		frame = np.roll(frame, shift=1440, axis=1)
		return frame

	def transform_3(self, frame):
		return frame

	def transform_4(self, frame):
		frame = np.flip(frame, axis=1) # flip azimuth
		frame = np.flip(frame, axis=0) # flip elevation
		return frame

	def transform_5(self, frame):
		frame = np.roll(frame, shift=480, axis=1) # roll by pi/2
		frame = np.flip(frame, axis=0)
		return frame

	def transform_6(self, frame):
		frame = np.flip(frame, axis=1)
		frame = np.roll(frame, shift=480, axis=1) # roll by pi/2
		return frame

	def transform_7(self, frame):
		frame = np.roll(frame, shift=960, axis=1)
		return frame

	def transform_8(self, frame):
		frame = np.flip(frame, axis=1)
		frame = np.roll(frame, shift=960, axis=1)
		frame = np.flip(frame, axis=0)
		return frame

	def process_video(self, output_path, transformation_id):
		out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.frame_width, self.frame_height))

		transformation_func = self.apply_transformation(transformation_id)

		while True:
			ret, frame = self.cap.read()
			if not ret:
				break

			transformed_frame = transformation_func(frame)
			transformed_frame = cv2.convertScaleAbs(transformed_frame)
			out.write(transformed_frame)

		self.cap.release()
		out.release()
		cv2.destroyAllWindows()

# Example usage:
video_path = "/Users/adrianromanguzman/Downloads/video_dev/dev-train-sony/fold3_room22_mix011.mp4"
output_path = "transformed_video_tranform1.mp4"
transformation_id = 1  # Replace with the desired transformation ID
transformer = VideoPixelTransformer(video_path)
transformer.process_video(output_path, transformation_id)
output_path = "transformed_video_tranform2.mp4"
transformation_id = 2  # Replace with the desired transformation ID
transformer = VideoPixelTransformer(video_path)
transformer.process_video(output_path, transformation_id)
output_path = "transformed_video_tranform5.mp4"
transformation_id = 5  # Replace with the desired transformation ID
transformer = VideoPixelTransformer(video_path)
transformer.process_video(output_path, transformation_id)
output_path = "transformed_video_tranform6.mp4"
transformation_id = 6  # Replace with the desired transformation ID
transformer = VideoPixelTransformer(video_path)
transformer.process_video(output_path, transformation_id)
