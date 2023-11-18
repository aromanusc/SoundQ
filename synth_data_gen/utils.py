import math

# Reference METU outter trayectory:  bottom outter trayectory
REF_OUT_TRAJ = ["034", "024", "014", "004", "104", "204",
			"304", "404", "504", "604", "614", "624",
			"634", "644", "654", "664", "564", "464",
			"364", "264", "164", "064", "054", "044"]
# Reference METU inner trayectory:  bottom inner trayectory
REF_IN_TRAJ = ["134", "124", "114", "214","314", "414", "514", "524",
				"534", "544", "554", "454", "354", "254", "154", "145"]

def get_mic_xyz():
	"""
	Get em32 microphone coordinates in 3D space
	"""
	return [(3 - 3) * 0.5, (3 - 3) * 0.5, (2 - 2) * 0.3 + 1.5]

def az_ele_from_source_radians(ref_point, src_point):
	"""
	Calculates the azimuth and elevation between a reference point and a source point in 3D space
	Args:
		ref_point (list): A list of three floats representing the x, y, and z coordinates of the reference point
		src_point (list): A list of three floats representing the x, y, and z coordinates of the other point
	Returns:
		A tuple of two floats representing the azimuth and elevation angles in radians plus distance between reference and source point
	"""
	dx = src_point[0] - ref_point[0]
	dy = src_point[1] - ref_point[1]
	dz = src_point[2] - ref_point[2]
	azimuth = math.atan2(dy, dx)
	distance = math.sqrt(dx**2 + dy**2 + dz**2)
	elevation = math.asin(dz/distance)
	return azimuth, elevation, distance


def az_ele_from_source(ref_point, src_point):
	"""
	Calculates the azimuth and elevation between a reference point and a source point in 3D space.

	Args:
		ref_point (list): A list of three floats representing the x, y, and z coordinates of the reference point.
		src_point (list): A list of three floats representing the x, y, and z coordinates of the other point.

	Returns:
		A tuple of two floats representing the azimuth and elevation angles in degrees plus the distance between the reference and source points.
	"""
	dx = src_point[0] - ref_point[0]
	dy = src_point[1] - ref_point[1]
	dz = src_point[2] - ref_point[2]

	azimuth = math.degrees(math.atan2(dy, dx))
	distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
	elevation = math.degrees(math.asin(dz / distance))

	return azimuth, elevation, distance
