import os
import cv2
import time
import numpy as np
from argparse import ArgumentParser

IPD = 6.5
MONITOR_W = 38.5
NEAR_DIST = 5
FAR_DIST = 70


def generate_stereo(in_dir, depth_dir, depth_prefix, out_dir, filename):
    print("=== Start processing:", filename, "===")
    left = cv2.imread(os.path.join(in_dir, filename + ".jpg"))
    depth_src = cv2.imread(os.path.join(depth_dir, depth_prefix + filename + ".png"))

    start_time = time.time()

    h, w, c = left.shape
    print("w:", w, "h:", h, ", c:", c)

    depth = cv2.cvtColor(depth_src, cv2.COLOR_BGR2GRAY)
    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min)

    right = np.zeros_like(left)

    deviation_cm = IPD * NEAR_DIST / FAR_DIST
    deviation = deviation_cm * w / MONITOR_W
    print("deviation:", deviation)

    for row in range(h):
        for col in range(w):
            col_r = col - int((1 - depth[row][col]) * deviation)
            if col_r >= 0:
                right[row][col_r] = left[row][col]

    print(time.time() - start_time, "seconds for base generation")
    start_time = time.time()

    right_fix = np.array(right)
    gray = cv2.cvtColor(right_fix, cv2.COLOR_BGR2GRAY)
    rows, cols = np.where(gray == 0)
    cnt = 0
    for row, col in zip(rows, cols):
        cnt = cnt + 1
        for offset in range(1, int(deviation)):
            r_offset = col + offset
            l_offset = col - offset
            if r_offset < w and not np.all(right_fix[row][r_offset] == 0):
                right_fix[row][col] = right_fix[row][r_offset]
                break
            if l_offset >= 0 and not np.all(right_fix[row][l_offset] == 0):
                right_fix[row][col] = right_fix[row][l_offset]
                break
    print(cnt, "interpolated points")

    print(time.time() - start_time, "seconds for interpolation")
    result = np.hstack([left, right_fix])
    cv2.imwrite(os.path.join(out_dir, "stereo_" + filename + ".jpg"), result)
    # cv2.imshow("preview", result)
    # cv2.waitKey(0)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-i", "--in-dir", help="Input directory for source images", default="example")
    arg_parser.add_argument("-d", "--depth-dir", help="Directory of depth maps", default="depth")
    arg_parser.add_argument("-p", "--depth-prefix", help="Prefix of file name for depth maps", default="MiDaS_")
    arg_parser.add_argument("-o", "--out-dir", help="Output directory for stereo images", default="stereo")
    args = arg_parser.parse_args()

    in_dir = args.in_dir
    depth_dir = args.depth_dir
    depth_prefix = args.depth_prefix
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for f in os.listdir(in_dir):
        filename = f.split(".")[0]
        generate_stereo(in_dir, depth_dir, depth_prefix, out_dir, filename)


if __name__ == "__main__":
    main()
