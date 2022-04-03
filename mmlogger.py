from logging import getLogger, INFO, StreamHandler, FileHandler, lastResort

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)


class PoseLogger:
    def __init__(self):
        fh = FileHandler(f"pose_" + {time.time} + ".csv")
        logger.addHandler(fh)  # 日ごとのCSVファイルハンドラを追加

    def on_message(self, msg):
        logger.info("{}".format(msg))
