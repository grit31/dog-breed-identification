# import os
# import pandas as pd
#
# root_dir = r"D:\自学资料\C++\施磊C++全套\全套资料\施磊资源\C++ 施磊\01-高级】C++全套数据结构算法-进阶高级开发必备-大厂面试必备"
#
# data = []
#
# for folder in os.listdir(root_dir):
#     folder_path = os.path.join(root_dir, folder)
#     if os.path.isdir(folder_path):
#         for file in os.listdir(folder_path):
#             if file.lower().endswith('.ts'):  # 只统计ts视频文件
#                 data.append([folder, file])
#
# df = pd.DataFrame(data, columns=["文件夹名", "子文件名"])
# df.to_excel("目录结构展开_仅视频.xlsx", index=False)


#
# import os
# import pandas as pd
# from moviepy.editor import VideoFileClip
#
# def get_video_duration(file_path):
#     try:
#         with VideoFileClip(file_path) as clip:
#             return clip.duration  # 单位：秒
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return 0
#
#
# root_dir = r"D:\自学资料\C++\施磊C++全套\全套资料\施磊资源\C++ 施磊\01-高级】C++全套数据结构算法-进阶高级开发必备-大厂面试必备"
#
# data = []
#
# for folder in os.listdir(root_dir):
#     folder_path = os.path.join(root_dir, folder)
#     if os.path.isdir(folder_path):
#         files = [f for f in os.listdir(folder_path) if f.lower().endswith('.ts')]
#         files.sort()  # 可选：按文件名排序
#         total_duration = 0
#         for file in files:
#             file_path = os.path.join(folder_path, file)
#             duration = get_video_duration(file_path)
#             total_duration += duration
#             data.append([folder, file, duration, total_duration])
#
# df = pd.DataFrame(data, columns=["文件夹名", "子文件名", "视频时长（秒）", "累计时长（秒）"])
#
# # 可选：把秒转为“hh:mm:ss”格式
# def format_time(seconds):
#     h = int(seconds // 3600)
#     m = int((seconds % 3600) // 60)
#     s = int(seconds % 60)
#     return f"{h:02d}:{m:02d}:{s:02d}"
#
# df["视频时长"] = df["视频时长（秒）"].apply(format_time)
# df["累计时长"] = df["累计时长（秒）"].apply(format_time)
#
# # 导出excel
# df.to_excel("视频目录与时长统计.xlsx", index=False)



import os
import pandas as pd
from moviepy.editor import VideoFileClip

def get_video_duration(file_path):
    try:
        with VideoFileClip(file_path) as clip:
            return clip.duration  # 秒
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return 0

root_dir = r"D:\自学资料\C++\施磊C++全套\全套资料\施磊资源\C++ 施磊\01-高级】C++全套数据结构算法-进阶高级开发必备-大厂面试必备"

data = []
total_duration = 0  # 全局累计时长

# 建一个所有视频文件的全路径清单
video_files = []
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.ts')]
        files.sort()
        for file in files:
            file_path = os.path.join(folder_path, file)
            video_files.append((folder, file, file_path))

for folder, file, file_path in video_files:
    duration = get_video_duration(file_path)
    total_duration += duration
    data.append([folder, file, duration, total_duration])

df = pd.DataFrame(data, columns=["文件夹名", "子文件名", "视频时长（秒）", "累计时长（秒）"])

# 可选：把秒转为“hh:mm:ss”格式
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

df["视频时长"] = df["视频时长（秒）"].apply(format_time)
df["累计时长"] = df["累计时长（秒）"].apply(format_time)

df.to_excel("全局累计时长统计.xlsx", index=False)
