# -*- coding: utf-8 -*-
"""
Complimentary Machine - Pi 5 Final Art Version (RunPod Pods)
修改：
1. API 调用改为 RunPod Pods FastAPI 格式
2. 使用 multipart/form-data 发送图片
3. 不需要 RunPod API Key
"""

import os
import sys
import time
import curses
import threading
import queue
import cv2
import requests
import io
import math
import textwrap
import numpy as np
import subprocess
from PIL import Image

# --- Configuration ---
# RunPod Pods 配置
# 格式: https://{POD_ID}-{PORT}.proxy.runpod.net
# 例如: https://abc123def-8000.proxy.runpod.net
POD_URL = os.environ.get("POD_URL", "https://YOUR_POD_ID-8000.proxy.runpod.net")
API_ENDPOINT = f"{POD_URL}/compliment"

VIDEO_FILE_PATH = "loop.mov" 

# TTS 配置
PIPER_BINARY = "/home/yanqihe/piper/piper"
VOICE_MODEL = "/home/yanqihe/piper/voice_danny.onnx" 

# 字符集
ASCII_CHARS = np.array(list(" .:-!ilItconepLTCOPENM"))

CAMERA_ID = 0 
POST_TTS_DELAY = 0  # 语音播完后等待多久再抓下一帧（秒），设为0则立刻抓
MAX_HISTORY = 1 

# Global State
stop_event = threading.Event()
display_queue = queue.Queue()
tts_queue = queue.Queue()
tts_ready_event = threading.Event()  # TTS播放完成信号
tts_ready_event.set()  # 初始状态：可以开始 

# Buffers
current_frame_data = []      
current_video_loop_data = [] 
current_frame_img = None
img_lock = threading.Lock()

# Layout State
layout_state = {
    "left_w": 40,  
    "left_h": 30,  
    "right_w": 40, 
    "right_h": 20, 
}

last_interaction_time = time.time()
is_inferencing = False
bot_response_history = ["Waiting for presence..."] 

# --- 1. Camera Processing (Left Side - 8 Color Cover) ---
def frame_to_ascii_color_numpy_cover(frame, target_cols, target_rows):
    if target_cols < 1: target_cols = 1
    if target_rows < 1: target_rows = 1
    
    h_src, w_src = frame.shape[:2]
    char_aspect = 2.0
    target_ratio = target_cols / (target_rows * char_aspect)
    src_ratio = w_src / h_src

    crop_w, crop_h = w_src, h_src
    if target_ratio < src_ratio:
        crop_h = h_src
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = w_src
        crop_h = int(crop_w / target_ratio)
        
    start_x = (w_src - crop_w) // 2
    start_y = (h_src - crop_h) // 2
    if start_x < 0: start_x = 0
    if start_y < 0: start_y = 0
    crop_w = min(crop_w, w_src - start_x)
    crop_h = min(crop_h, h_src - start_y)
    
    roi = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
    resized = cv2.resize(roi, (target_cols, target_rows))
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    indices = (gray / 255 * (len(ASCII_CHARS) - 1)).astype(int)
    char_matrix = ASCII_CHARS[indices]

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    color_matrix = np.full((target_rows, target_cols), 7, dtype=int)
    
    mask_very_dark = (v < 60)
    mask_low_sat = (s < 60) & (~mask_very_dark)
    
    mask_red = ((h < 15) | (h > 165))
    mask_yellow = (h >= 15) & (h < 45)
    mask_green = (h >= 45) & (h < 75)
    mask_cyan = (h >= 75) & (h < 105)
    mask_blue = (h >= 105) & (h < 135)
    mask_magenta = (h >= 135) & (h < 165)

    color_matrix[mask_red] = 1
    color_matrix[mask_yellow] = 3
    color_matrix[mask_green] = 2
    color_matrix[mask_cyan] = 6
    color_matrix[mask_blue] = 4
    color_matrix[mask_magenta] = 5
    
    color_matrix[mask_low_sat] = 7 
    color_matrix[mask_very_dark] = 8 

    render_data = []
    for i in range(target_rows):
        render_data.append(("".join(char_matrix[i]), color_matrix[i]))
    return render_data

def try_open_camera(max_retries=30, retry_delay=2.0):
    """
    尝试打开摄像头，带重试机制。
    解决 Pi5 启动时 USB 摄像头可能未就绪的问题。
    """
    for attempt in range(max_retries):
        if stop_event.is_set():
            return None
            
        for index in [0, 1, 2]:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"📷 摄像头已连接 (index={index}, 尝试 {attempt + 1}/{max_retries})")
                    return cap
                cap.release()
        
        if attempt < max_retries - 1:
            print(f"⏳ 等待摄像头就绪... ({attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
    
    print("❌ 无法连接摄像头，请检查 USB 连接")
    return None

def camera_thread_func():
    global current_frame_data, current_frame_img
    
    cap = try_open_camera(max_retries=30, retry_delay=2.0)
    
    if not cap or not cap.isOpened():
        current_frame_data = [("CAMERA NOT FOUND - CHECK USB", np.array([1] * 30))]
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(1.0)
    
    consecutive_failures = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            consecutive_failures = 0
            frame = cv2.flip(frame, 1)
            t_w = layout_state["left_w"]
            t_h = layout_state["left_h"]
            current_frame_data = frame_to_ascii_color_numpy_cover(frame, target_cols=t_w, target_rows=t_h)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with img_lock:
                current_frame_img = Image.fromarray(rgb_frame)
        else:
            consecutive_failures += 1
            if consecutive_failures > 30:
                print("⚠️ 摄像头断开，尝试重新连接...")
                cap.release()
                time.sleep(1.0)
                cap = try_open_camera(max_retries=10, retry_delay=2.0)
                if not cap:
                    current_frame_data = [("CAMERA DISCONNECTED", np.array([1] * 20))]
                    break
                consecutive_failures = 0
            time.sleep(0.1)
        time.sleep(0.03)
    
    if cap:
        cap.release()

# --- 2. Video Loop ---
def auto_crop_frame(frame):
    if frame is None: return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    rows = cv2.reduce(thresh, 1, cv2.REDUCE_MAX).reshape(-1)
    cols = cv2.reduce(thresh, 0, cv2.REDUCE_MAX).reshape(-1)
    y_indices = np.where(rows > 0)[0]
    x_indices = np.where(cols > 0)[0]
    if len(y_indices) == 0 or len(x_indices) == 0: return frame
    y1, y2 = y_indices[0], y_indices[-1] + 1
    x1, x2 = x_indices[0], x_indices[-1] + 1
    return frame[y1:y2, x1:x2]

def frame_to_ascii_mono_black_ink(frame, target_width, max_height):
    height, original_width = frame.shape[:2]
    aspect_ratio_correction = 2.0 
    aspect_ratio = height / original_width / aspect_ratio_correction
    calc_height = int(target_width * aspect_ratio)
    final_width = target_width
    final_height = calc_height
    if calc_height > max_height and max_height > 0:
        final_height = max_height
        final_width = int(final_height / aspect_ratio)
    if final_width < 1: final_width = 1
    if final_height < 1: final_height = 1
    resized = cv2.resize(frame, (final_width, final_height))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    indices = ((1.0 - gray / 255) * (len(ASCII_CHARS) - 1)).astype(int)
    char_matrix = ASCII_CHARS[indices]
    rows = []
    for i in range(final_height):
        rows.append("".join(char_matrix[i]))
    return rows

def video_player_thread_func():
    global current_video_loop_data
    if not os.path.exists(VIDEO_FILE_PATH):
        current_video_loop_data = ["VIDEO FILE NOT FOUND"]
        return
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    delay = 1.0 / fps
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        cropped_frame = auto_crop_frame(frame)
        avail_w = max(10, layout_state["right_w"] - 2)
        avail_h = max(5, layout_state["right_h"])
        current_video_loop_data = frame_to_ascii_mono_black_ink(cropped_frame, target_width=avail_w, max_height=avail_h)
        time.sleep(delay)
    cap.release()

# --- 3. TTS Thread ---
def tts_thread_func():
    while not stop_event.is_set():
        try:
            text = tts_queue.get(timeout=1.0)
            
            piper_cmd = [PIPER_BINARY, "--model", VOICE_MODEL, "--output-raw"]
            aplay_cmd = ["aplay", "-r", "16000", "-f", "S16_LE", "-t", "raw", "-c", "1", "-q"] 
            
            try:
                piper_proc = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, bufsize=0)
                aplay_proc = subprocess.Popen(aplay_cmd, stdin=piper_proc.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=0)
                
                piper_proc.stdout.close()
                
                if not text.endswith("\n"): text += "\n"
                piper_proc.stdin.write(text.encode("utf-8"))
                piper_proc.stdin.flush()
                piper_proc.stdin.close()
                
                piper_proc.wait()
                aplay_proc.wait()
            except Exception as e:
                pass
            finally:
                tts_ready_event.set()
            
            tts_queue.task_done()
        except queue.Empty:
            pass

# --- 4. API & Logic (RunPod Pods - FastAPI) ---
def call_pod_api(image, user_text=None):
    """
    调用 RunPod Pods 上的 FastAPI 服务
    使用 multipart/form-data 发送图片
    """
    try:
        # 将 PIL Image 转为 bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=60)
        img_byte_arr.seek(0)
        
        # 构建 multipart/form-data 请求
        files = {
            'image': ('image.jpg', img_byte_arr, 'image/jpeg')
        }
        
        data = {}
        if user_text:
            data['user_text'] = user_text
        
        # 发送请求
        resp = requests.post(API_ENDPOINT, files=files, data=data, timeout=30)
        
        if resp.status_code == 200:
            result = resp.json()
            if "error" in result:
                return f"API Error: {result['error']}"
            return result.get("compliment", "No response")
        else:
            return f"HTTP Error: {resp.status_code}"
            
    except requests.exceptions.Timeout:
        return "Timeout"
    except requests.exceptions.ConnectionError:
        return "Connection Error - Is Pod running?"
    except Exception as e:
        return f"Error: {str(e)}"

def worker_func():
    global last_interaction_time, is_inferencing
    
    # 启动后先等一小段时间让摄像头稳定
    time.sleep(2.0)
    
    while not stop_event.is_set():
        # 等待 TTS 播放完成（阻塞等待，播完立刻继续）
        tts_ready_event.wait()
        
        # 可选：播完后加一个小延迟，让观众有喘息时间
        time.sleep(POST_TTS_DELAY)
        
        is_inferencing = True
        last_interaction_time = time.time()
        
        img = None
        with img_lock:
            if current_frame_img: img = current_frame_img.copy()
        
        if img:
            resp = call_pod_api(img, None)
            if "Error" not in resp and "Timeout" not in resp:
                tts_ready_event.clear()
                bot_response_history.clear()
                bot_response_history.append(resp)
                tts_queue.put(resp)
        
        is_inferencing = False
        time.sleep(0.1)

# --- 5. UI ---
def draw_ui(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(40)
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors() 
        
        for i in range(1, 8): curses.init_pair(i, i, curses.COLOR_BLACK)
        curses.init_pair(8, curses.COLOR_BLACK, -1) 
        curses.init_pair(20, curses.COLOR_BLACK, curses.COLOR_WHITE)

    while not stop_event.is_set():
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        
        mid_x = width // 2
        right_w = width - mid_x
        
        full_text_lines = []
        text_width = right_w - 6
        for resp in bot_response_history:
            lines = textwrap.wrap(resp, width=text_width)
            full_text_lines.extend(lines)
            
        max_text_h = height // 3
        if len(full_text_lines) > max_text_h:
            full_text_lines = full_text_lines[-max_text_h:]
            
        text_area_h = len(full_text_lines) + 4
        video_avail_height = height - text_area_h
        
        layout_state["left_w"] = mid_x
        layout_state["left_h"] = height
        layout_state["right_w"] = right_w
        layout_state["right_h"] = video_avail_height
        
        # 1. Left (Camera)
        cam_h = min(len(current_frame_data), height)
        for i in range(cam_h):
            row_chars, row_colors = current_frame_data[i]
            display_chars = row_chars[:mid_x]
            display_colors = row_colors[:mid_x]
            for j, char in enumerate(display_chars):
                try:
                    c_idx = display_colors[j]
                    attr = 0
                    if c_idx == 8: 
                        attr = curses.color_pair(8) | curses.A_BOLD
                    else:
                        attr = curses.color_pair(c_idx)
                    stdscr.addch(i, j, char, attr)
                except: pass

        # 2. Right (Video)
        right_bg_attr = curses.color_pair(20)
        for y in range(height):
            try: stdscr.addstr(y, mid_x, " " * right_w, right_bg_attr)
            except: pass

        vid_data = current_video_loop_data
        vid_h = len(vid_data)
        vid_start_y = max(0, (video_avail_height - vid_h) // 2)
        for i in range(min(vid_h, video_avail_height)):
            line = vid_data[i]
            padding_x = (right_w - len(line)) // 2
            try:
                stdscr.addstr(vid_start_y + i, mid_x + max(0, padding_x), line, right_bg_attr | curses.A_BOLD)
            except: pass

        # 3. Text
        text_start_y = height - len(full_text_lines) - 2
        try:
            stdscr.addstr(text_start_y - 2, mid_x + 2, "_" * (right_w - 4), right_bg_attr | curses.A_BOLD)
        except: pass
        
        for i, line in enumerate(full_text_lines):
            if text_start_y + i >= height: break
            try:
                stdscr.addstr(text_start_y + i, mid_x + 3, line, right_bg_attr)
            except: pass

        try:
            key = stdscr.getch()
            if key == 27: stop_event.set()
        except: pass
        
        stdscr.refresh()

def main():
    # 检查配置
    if "YOUR_POD_ID" in POD_URL:
        print("⚠️  请先配置 Pod URL!")
        print()
        print("   方法1: 设置环境变量:")
        print("     export POD_URL='https://abc123-8000.proxy.runpod.net'")
        print()
        print("   方法2: 直接修改脚本顶部的 POD_URL")
        print()
        print("   Pod URL 格式: https://{POD_ID}-{PORT}.proxy.runpod.net")
        print("   可在 RunPod Console → Pod 详情页 → Connect 找到")
        sys.exit(1)
    
    print(f"🚀 使用 RunPod Pod API: {API_ENDPOINT}")
    
    threading.Thread(target=camera_thread_func, daemon=True).start()
    threading.Thread(target=video_player_thread_func, daemon=True).start() 
    threading.Thread(target=tts_thread_func, daemon=True).start() 
    threading.Thread(target=worker_func, daemon=True).start()
    try:
        curses.wrapper(draw_ui)
    except KeyboardInterrupt: pass
    finally:
        stop_event.set()
        print("Bye.")

if __name__ == "__main__":
    main()
