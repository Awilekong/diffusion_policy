#!/usr/bin/env python3
"""
Crop 参数生成器 (视频版本)
==========================
通过鼠标框选的方式来生成 crop_pos 和 crop_shape 参数。
支持视频输入，使用滑动条选择帧。

使用方法:
    # 单个视频
    python tools/crop_param_generator.py --video /path/to/video.mp4

    # 多相机视频目录
    python tools/crop_param_generator.py --video_dir /path/to/videos --multi_camera

    # 自定义 resize 尺寸
    python tools/crop_param_generator.py --video /path/to/video.mp4 --resize 240 320

操作说明:
    - 滑动条: 选择视频帧
    - 鼠标左键拖动: 框选 crop 区域
    - 按 'r': 重置框选
    - 按 'c': 确认当前选择并继续下一个视频
    - 按 'q': 退出程序
    - 按 's': 保存当前配置到 YAML 文件
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import yaml


class CropSelector:
    def __init__(self, resize_shape=(240, 320)):
        """
        Args:
            resize_shape: (height, width) resize 后的图像尺寸
        """
        self.resize_shape = resize_shape  # (H, W)
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_rect = None
        self.confirmed_rects = {}  # {camera_name: (top, left, height, width)}

        # 视频相关
        self.current_frame_idx = 0
        self.total_frames = 0
        self.cap = None
        self.current_frame = None
        self.current_frame_resized = None

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            # 计算矩形参数
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            left = min(x1, x2)
            top = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            if width > 0 and height > 0:
                self.current_rect = (top, left, height, width)

    def trackbar_callback(self, val):
        """滑动条回调"""
        self.current_frame_idx = val
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, val)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                # BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                self.current_frame_resized = cv2.resize(
                    frame_rgb,
                    (self.resize_shape[1], self.resize_shape[0])
                )

    def draw_overlay(self, img):
        """在图像上绘制选择框和参数信息"""
        display = img.copy()
        # 转回 BGR 用于 OpenCV 显示
        display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        h, w = display.shape[:2]

        # 绘制当前正在拖动的矩形
        if self.start_point and self.end_point:
            cv2.rectangle(display, self.start_point, self.end_point, (0, 255, 0), 2)

        # 显示当前选择的参数
        if self.current_rect:
            top, left, height, width = self.current_rect
            # 绘制确认后的矩形（用不同颜色）
            cv2.rectangle(display, (left, top), (left + width, top + height), (0, 255, 255), 2)

            # 显示参数信息
            info_lines = [
                f"crop_pos: [{top}, {left}]  # [top, left]",
                f"crop_shape: [{height}, {width}]  # [height, width]",
                f"resize_shape: [{self.resize_shape[0]}, {self.resize_shape[1]}]",
                "",
                f"Region: ({left}, {top}) -> ({left+width}, {top+height})",
                f"Coverage: {width}x{height} / {w}x{h} = {100*width*height/(w*h):.1f}%"
            ]

            y_offset = 30
            for line in info_lines:
                cv2.putText(display, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20
        else:
            cv2.putText(display, "Drag mouse to select crop region", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示帧信息
        frame_info = f"Frame: {self.current_frame_idx}/{self.total_frames-1}"
        cv2.putText(display, frame_info, (w - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 显示操作提示
        tips = "[R]eset  [C]onfirm  [S]ave  [Q]uit  |  Slider: select frame"
        cv2.putText(display, tips, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return display

    def select_crop_region_from_video(self, video_path, camera_name=None):
        """
        从视频中交互式选择 crop 区域

        Args:
            video_path: 视频路径
            camera_name: 相机名称，用于多相机配置

        Returns:
            (top, left, height, width) 或 None
        """
        # 打开视频
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            print(f"Error: Cannot open video: {video_path}")
            return None

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        original_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 读取第一帧
        ret, frame = self.cap.read()
        if not ret:
            print(f"Error: Cannot read video: {video_path}")
            self.cap.release()
            return None

        self.current_frame = frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame_resized = cv2.resize(
            frame_rgb,
            (self.resize_shape[1], self.resize_shape[0])
        )

        # 显示信息
        window_name = f"Crop Selector - {camera_name or Path(video_path).name}"
        print(f"\n{'='*60}")
        print(f"Video: {video_path}")
        print(f"Original resolution: {original_w}x{original_h} (WxH)")
        print(f"Total frames: {self.total_frames}, FPS: {fps:.1f}")
        print(f"Duration: {self.total_frames/fps:.1f} seconds")
        print(f"Resized to: {self.resize_shape[1]}x{self.resize_shape[0]} (WxH)")
        print(f"{'='*60}")

        # 重置状态
        self.current_rect = None
        self.start_point = None
        self.end_point = None
        self.current_frame_idx = 0

        # 创建窗口和滑动条
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        cv2.createTrackbar('Frame', window_name, 0, self.total_frames - 1, self.trackbar_callback)

        while True:
            if self.current_frame_resized is not None:
                display = self.draw_overlay(self.current_frame_resized)
                cv2.imshow(window_name, display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('r'):  # Reset
                self.current_rect = None
                self.start_point = None
                self.end_point = None
                print("Reset selection")

            elif key == ord('c'):  # Confirm
                if self.current_rect:
                    if camera_name:
                        self.confirmed_rects[camera_name] = self.current_rect
                    print(f"\nConfirmed crop parameters:")
                    print(f"  crop_pos: [{self.current_rect[0]}, {self.current_rect[1]}]")
                    print(f"  crop_shape: [{self.current_rect[2]}, {self.current_rect[3]}]")
                    cv2.destroyWindow(window_name)
                    self.cap.release()
                    return self.current_rect
                else:
                    print("Please select a region first!")

            elif key == ord('s'):  # Save
                self.save_config()

            elif key == ord('q'):  # Quit
                cv2.destroyWindow(window_name)
                self.cap.release()
                return None

            # 左右箭头键控制帧
            elif key == 81 or key == ord('a'):  # Left arrow or 'a'
                new_idx = max(0, self.current_frame_idx - 1)
                cv2.setTrackbarPos('Frame', window_name, new_idx)
                self.trackbar_callback(new_idx)
            elif key == 83 or key == ord('d'):  # Right arrow or 'd'
                new_idx = min(self.total_frames - 1, self.current_frame_idx + 1)
                cv2.setTrackbarPos('Frame', window_name, new_idx)
                self.trackbar_callback(new_idx)
            # 大跳跃
            elif key == ord('w'):  # 前进 10 帧
                new_idx = min(self.total_frames - 1, self.current_frame_idx + 10)
                cv2.setTrackbarPos('Frame', window_name, new_idx)
                self.trackbar_callback(new_idx)
            elif key == ord('x'):  # 后退 10 帧
                new_idx = max(0, self.current_frame_idx - 10)
                cv2.setTrackbarPos('Frame', window_name, new_idx)
                self.trackbar_callback(new_idx)

        cv2.destroyWindow(window_name)
        self.cap.release()
        return self.current_rect

    def save_config(self, output_path="crop_config.yaml"):
        """保存配置到 YAML 文件"""
        if not self.confirmed_rects and not self.current_rect:
            print("No crop regions to save!")
            return

        config = {
            'resize_shape': list(self.resize_shape),
            'crop_pos': {},
            'crop_shape': {}
        }

        # 添加已确认的配置
        for camera_name, rect in self.confirmed_rects.items():
            top, left, height, width = rect
            config['crop_pos'][camera_name] = [top, left]
            config['crop_shape'][camera_name] = [height, width]

        # 如果有当前选择但未确认，也添加
        if self.current_rect and 'current' not in self.confirmed_rects:
            top, left, height, width = self.current_rect
            config['crop_pos']['current'] = [top, left]
            config['crop_shape']['current'] = [height, width]

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"\nConfig saved to: {output_path}")
        print(yaml.dump(config, default_flow_style=False))


def find_videos(path, extensions=('.mp4', '.avi', '.mov', '.mkv', '.webm')):
    """查找目录下的视频文件"""
    path = Path(path)
    if path.is_file():
        return [path]

    videos = []
    for ext in extensions:
        videos.extend(path.glob(f'*{ext}'))
        videos.extend(path.glob(f'*{ext.upper()}'))
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(
        description='Crop 参数生成器 (视频版本) - 通过鼠标框选生成 crop_pos 和 crop_shape',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个视频
  python tools/crop_param_generator.py --video /path/to/video.mp4

  # 多相机视频目录（自动检测 camera_0, camera_1, camera_2）
  python tools/crop_param_generator.py --video_dir /path/to/videos --multi_camera

  # 自定义 resize 尺寸
  python tools/crop_param_generator.py --video /path/to/video.mp4 --resize 240 320

操作说明:
  滑动条: 选择视频帧
  A/D 或 ←/→: 前后移动 1 帧
  W/X: 前后移动 10 帧
  鼠标左键拖动: 框选 crop 区域
  R: 重置框选
  C: 确认当前选择
  S: 保存配置到 YAML
  Q: 退出
        """
    )

    parser.add_argument('--video', type=str, help='单个视频路径')
    parser.add_argument('--video_dir', type=str, help='视频目录路径')
    parser.add_argument('--resize', type=int, nargs=2, default=[240, 320],
                       metavar=('HEIGHT', 'WIDTH'),
                       help='Resize 尺寸，与训练配置一致 (默认: 240 320)')
    parser.add_argument('--multi_camera', action='store_true',
                       help='多相机模式，按 camera_0, camera_1, camera_2 顺序处理')
    parser.add_argument('--output', type=str, default='crop_config.yaml',
                       help='输出配置文件路径 (默认: crop_config.yaml)')

    args = parser.parse_args()

    if not args.video and not args.video_dir:
        parser.print_help()
        print("\n错误: 请指定 --video 或 --video_dir")
        return

    resize_shape = tuple(args.resize)  # (H, W)
    selector = CropSelector(resize_shape=resize_shape)

    print(f"\n{'='*60}")
    print("Crop 参数生成器 (视频版本)")
    print(f"{'='*60}")
    print(f"Resize shape: {resize_shape[0]}x{resize_shape[1]} (HxW)")
    print(f"Output file: {args.output}")
    print(f"{'='*60}\n")

    if args.video:
        # 单个视频模式
        result = selector.select_crop_region_from_video(args.video)
        if result:
            top, left, height, width = result
            print("\n" + "="*60)
            print("最终配置 (复制到 YAML):")
            print("="*60)
            print(f"resize_shape: [{resize_shape[0]}, {resize_shape[1]}]")
            print(f"crop_shape: [{height}, {width}]")
            print(f"crop_pos: [{top}, {left}]  # [top, left]")

    elif args.video_dir:
        video_dir = Path(args.video_dir)

        if args.multi_camera:
            # 多相机模式
            camera_names = ['camera_0', 'camera_1', 'camera_2']
            for camera_name in camera_names:
                # 尝试查找该相机的视频
                patterns = [
                    f'{camera_name}*.mp4',
                    f'{camera_name}*.avi',
                    f'{camera_name}*.mov',
                    f'*{camera_name}*.mp4',
                    f'*{camera_name}*.avi',
                ]

                video_found = None
                for pattern in patterns:
                    matches = list(video_dir.glob(pattern))
                    if matches:
                        video_found = matches[0]
                        break

                if video_found:
                    print(f"\n处理 {camera_name}: {video_found}")
                    result = selector.select_crop_region_from_video(video_found, camera_name)
                    if result is None:
                        print("用户退出")
                        break
                else:
                    print(f"未找到 {camera_name} 的视频，跳过")

            # 输出最终配置
            if selector.confirmed_rects:
                print("\n" + "="*60)
                print("最终配置 (复制到 YAML):")
                print("="*60)
                print(f"resize_shape: [{resize_shape[0]}, {resize_shape[1]}]")
                print("crop_shape:")
                for cam, rect in selector.confirmed_rects.items():
                    print(f"  {cam}: [{rect[2]}, {rect[3]}]")
                print("crop_pos:")
                for cam, rect in selector.confirmed_rects.items():
                    print(f"  {cam}: [{rect[0]}, {rect[1]}]  # [top, left]")

                selector.save_config(args.output)
        else:
            # 普通目录模式，处理所有视频
            videos = find_videos(video_dir)
            if not videos:
                print(f"未在 {video_dir} 找到视频文件")
                return

            print(f"找到 {len(videos)} 个视频")
            for video_path in videos:
                result = selector.select_crop_region_from_video(video_path, video_path.stem)
                if result is None:
                    break

            if selector.confirmed_rects:
                selector.save_config(args.output)


if __name__ == '__main__':
    main()
