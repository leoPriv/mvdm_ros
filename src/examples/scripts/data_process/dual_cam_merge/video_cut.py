import ffmpeg
import subprocess

def extract_first_minute(input_path, output_path):

    try:
        # 检查FFmpeg是否可用
        try:
            subprocess.run(["ffmpeg", "-version"], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("FFmpeg未安装或未在系统路径中找到，请先安装FFmpeg")

        # 获取视频信息
        probe = ffmpeg.probe(input_path)
        video_info = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None
        )

        if not video_info:
            raise ValueError("输入文件中未找到视频流")

        # 构建FFmpeg命令
        input_stream = ffmpeg.input(input_path)

        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                # 关键参数 - 保持原始质量
                c='copy',               # 流复制模式(不重新编码)
                map='0',                # 包含所有原始流
                ss='00:00:00',         # 开始时间
                t='00:00:23',          # 持续时间1分钟
                avoid_negative_ts='make_zero',
                **{'movflags': '+faststart'}  # 优化网络播放
            )
            .global_args('-loglevel', 'error')  # 只显示错误信息
            .overwrite_output()
            .run()
        )

        print(f"成功提取前1分钟视频: {output_path}")
        return True

    except ffmpeg.Error as e:
        print(f"FFmpeg处理失败: {e.stderr.decode('utf8')}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    return False

# 使用示例
if __name__ == "__main__":

    input_video = 'camera1_20250702_225030-1min.mp4'
    output_video = 'camera1_20250702_225030-23s.mp4'

    if extract_first_minute(input_video, output_video):
        print("操作成功完成")
    else:
        print("操作失败")

